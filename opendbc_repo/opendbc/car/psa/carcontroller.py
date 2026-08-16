from opendbc.can.packer import CANPacker
from opendbc.car import Bus, DT_CTRL, structs, make_tester_present_msg
from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.psa.psacan import (create_lka_steering, create_resume_acc, create_disable_radar, create_enable_radar,
                                    create_HS2_DYN1_MDD_ETAT_2B6, create_HS2_DYN_MDD_ETAT_2F6)
from opendbc.car.psa.values import CarControllerParams
from opendbc.car.vehicle_model import VehicleModel
from numpy import interp
import math

try:
  from cereal import messaging
  sm = messaging.SubMaster(['modelV2'], poll='modelV2')
except ImportError:
  # cereal is only available in openpilot, not in standalone opendbc
  sm = None

LongCtrlState = structs.CarControl.Actuators.LongControlState

# pandad leaves the panda in ELM327 mode until card publishes CarParams, measured
# at 20-174 ms after the first control frame across 20 logged drives. The radar
# knockout goes to a diagnostic address, which ELM327 lets through, but the
# emulated 0x2B6/0x2F6 that take the radar's place are rejected until the switch
# to the PSA safety mode lands. Knocking the radar out inside that window leaves
# the ADAS bus silent, and the ESP (UC_FREIN) marks its ACC fields invalid after
# ~150 ms of silence, which openpilot then reports as accFaulted. Hold off well
# past the worst measured delay before touching the radar.
RADAR_DISABLE_FRAME = 100  # 1.0 s

# The radar resumed 220-230 ms after the programming session ended in the logs, but that
# was via the S3 timeout, so allow generous margin. The release ends as soon as the real
# 0x2B6 reappears, so this only bounds the case where it never does.
RADAR_ENABLE_TIMEOUT_FRAMES = 200  # 2.0 s


def get_safety_CP():
  # We use the PSA_PEUGEOT_208 platform for lateral limiting to match safety
  from opendbc.car.psa.interface import CarInterface
  return CarInterface.get_non_essential_params("PSA_PEUGEOT_208")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.main])
    self.apply_angle_last = 0
    self.lat_active_last = False
    self.engage_frame = 0
    self.radar_disable_sent = False
    self.radar_disabled = False
    self.radar_release_requested = False
    self.radar_released = False
    self.radar_release_frame = 0
    self.status = 2
    self.bars = 4

    # Vehicle model used for lateral limiting
    self.VM = VehicleModel(get_safety_CP())

  def release_radar(self) -> bool:
    """Put the radar ECU back on the ADAS bus, called once alpha long is switched off.

    Dropping straight off the bus leaves the radar in its programming session until the
    S3 timer expires ~5 s later, and the ESP (UC_FREIN) marks its ACC fields invalid
    after ~150 ms of that, which openpilot reports as accFaulted. This has to run while
    the car is still onroad: pandad puts the panda in NO_OUTPUT the moment deviceState
    goes offroad, and everything we send after that is rejected.

    Non-blocking, the work happens in update(). Returns True once the radar is back.
    """
    if not self.CP.openpilotLongitudinalControl:
      return True  # nothing was ever knocked out
    self.radar_release_requested = True
    return self.radar_released

  def update(self, CC, CS, now_nanos, starpilot_toggles):
    can_sends = []
    actuators = CC.actuators

    # lateral control
    apply_angle = actuators.steeringAngleDeg

    # smooth engagement: blend from current wheel angle to commanded angle over ~1.0s
    # on rising edge of latActive to avoid a lunge when the initial command is far from the wheel
    ENGAGE_FRAMES = 100
    if CC.latActive and not self.lat_active_last:
      self.engage_frame = 0
    if CC.latActive and self.engage_frame < ENGAGE_FRAMES:
      self.engage_frame += 1
      blend = self.engage_frame / ENGAGE_FRAMES
      apply_angle = blend * apply_angle + (1 - blend) * CS.out.steeringAngleDeg
    self.lat_active_last = CC.latActive

    # low-pass filter at low speeds to suppress high-frequency jitter at low speeds.
    # the reason for this happening is unknown, but it may be related to the EPS torque sensor noise or quantization.
    # the filter time constant is reduced as speed increases to avoid excessive delay at higher speeds.
    if CC.latActive and CS.out.vEgoRaw < 5.0:
      tau = interp(CS.out.vEgoRaw, [0.5, 5.0], [0.3, 0.1])
      alpha = 1 - math.exp(-DT_CTRL / tau)
      apply_angle = alpha * apply_angle + (1 - alpha) * self.apply_angle_last

    apply_angle = apply_steer_angle_limits_vm(apply_angle, self.apply_angle_last, CS.out.vEgoRaw,
                                              CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, self.VM)

    # EPS disengages on steering override, activation sequence 2->3->4 to re-engage
    # STATUS  -  0: UNAVAILABLE, 1: UNSELECTED, 2: READY, 3: AUTHORIZED, 4: ACTIVE
    if not CC.latActive:
      self.status = 2
    elif not CS.eps_active and not CS.out.steeringPressed:
      self.status = 2 if self.status == 4 else self.status + 1
    else:
      self.status = 4

    # TUNING
    # >=-0.5: Engine brakes only
    # <-0.5: Add friction brakes
    pitch = CC.orientationNED[1] if len(CC.orientationNED) == 3 else 0.0
    accel_slope = math.sin(pitch) * 9.81
    accel_cmd = actuators.accel + accel_slope

    brake_accel = -0.5

    # torque lookup
    ACCEL_LOOKUP = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    TORQUE_LOOKUP = [-400, -300, 120, 350, 550, 800, 1000]

    # calculate Torque
    torque_nm = interp(accel_cmd, ACCEL_LOOKUP, TORQUE_LOOKUP)
    torque = max(-400, min(torque_nm, 1000))

    braking = accel_cmd < brake_accel and not CS.out.gasPressed
    if self.CP.openpilotLongitudinalControl:
      if CC.hudControl.leadVisible:
        sm.update(0)
        leads_v3 = sm['modelV2'].leadsV3
        if leads_v3 and leads_v3[0].x:
          r = leads_v3[0].x[0] / (3 + CS.out.vEgo)
          if self.bars > 3:  # initialize from "no lead"
            self.bars = min(3, int(r))
          elif r > self.bars + 1.2:
            self.bars = min(3, self.bars + 1)
          elif r < self.bars - 0.2:
            self.bars = max(0, self.bars - 1)
      else:
        self.bars = 4

      if self.radar_release_requested:
        # hand the bus back, see release_radar()
        if not self.radar_released:
          if not self.radar_disable_sent:
            self.radar_released = True  # never knocked it out, nothing to hand back
          else:
            if self.radar_release_frame == 0:
              can_sends.append(create_enable_radar())
            self.radar_release_frame += 1
            self.radar_released = CS.radar_alive or self.radar_release_frame >= RADAR_ENABLE_TIMEOUT_FRAMES
      # disable radar ECU by setting to programming mode, see RADAR_DISABLE_FRAME
      elif not self.radar_disabled:
        if not self.radar_disable_sent and self.frame >= RADAR_DISABLE_FRAME:
          can_sends.append(create_disable_radar())
          self.radar_disable_sent = True
        # only start emulating once the real radar has actually gone quiet, two
        # ECUs transmitting 0x2B6 at once would collide on the bus
        self.radar_disabled = self.radar_disable_sent and not CS.radar_alive
      elif self.frame % 100 == 0:
        # keep radar ECU disabled by sending tester present
        can_sends.append(make_tester_present_msg(0x6b6, 1, suppress_response=False))

      # Highest torque seen without gas input: ~1000
      # Lowest torque seen without break mode: -560 (but only when transitioning from brake to accel mode, else -248)
      # Lowest brake mode accel seen: -4.85m/s²

      # stand in for the radar for exactly as long as it is off the bus
      if self.radar_disabled and not self.radar_released and self.frame % 2 == 0:
        can_sends.append(create_HS2_DYN1_MDD_ETAT_2B6(self.packer, self.frame // 2, actuators.accel, CS.out.cruiseState.enabled,
                                                      CS.out.gasPressed, braking, CS.out.brakePressed, CS.out.standstill, torque))
        can_sends.append(create_HS2_DYN_MDD_ETAT_2F6(self.packer, braking, CC.hudControl.leadVisible, self.bars))

    # stock long
    # emulate resume button every 3 seconds to prevent autohold timeout
    elif CC.latActive and CS.out.standstill and CC.hudControl.leadVisible:
      # map: {frame:status} - 0, 1
      status = {0: 0, 5: 1}.get(self.frame % 300)
      if status is not None:
        msg = CS.hs2_dat_mdd_cmd_452
        counter = (msg['COUNTER'] + 1) % 16
        can_sends.append(create_resume_acc(self.packer, counter, status, msg))

    can_sends.append(create_lka_steering(self.packer, CC.latActive, apply_angle, self.status, CS.lka_drive_mode))
    self.apply_angle_last = apply_angle

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = apply_angle
    self.frame += 1
    return new_actuators, can_sends
