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

# The ESP holds the car at a standstill for the radar (autohold-style) and reports it as
# ARRET_VHL_ADAS in 0x32D, about a second after the car comes to rest. What releases that
# hold is the launch pattern in 0x2B6, not the 0x2F6 pulse: across the stock drive-aways
# the ESP cleared ARRET_VHL_ADAS 70-100 ms after the first launch frame, then ramped the
# brakes out over ~0.5 s, and one stock launch carried no DRIVE_AWAY_REQUEST at all. The
# pulse is kept because the stock radar usually sends one, but only for its measured
# length of ~40 ms — see create_HS2_DYN1_MDD_ETAT_2B6.
DRIVE_AWAY_FRAMES = 4   # 40 ms at 100 Hz
LAUNCH_COMPLETE_SPEED = 0.5  # m/s, decel request released above this

# Torque floor while the ESP is still holding the car. The stock radar opened its launches
# with 458-939 N.m (six drive-aways) and advertised 700-1000 N.m of potential wheel torque
# throughout the hold; the accel map alone asks ~320 N.m at the planner's first positive
# command, below anything the stock radar was seen to use to get the car moving.
LAUNCH_TORQUE = 600  # N.m

# The planner's cruise source is a plain P controller on speed error clipped at
# A_CRUISE_MIN, so a 4.3 km/h set-speed step already saturates it at -1.2 m/s², and its own
# jerk limiter runs while disengaged too — measured pinned at -1.20 for the whole second
# before an engage frame, so engaging above the set speed landed the full request in one
# frame (-1.22 commanded, -1.46 m/s² achieved 0.8 s later). PSA has no jerk or gradient
# signal to hand the ESP the way Tesla, VW and Hyundai do, so the request is shaped here
# instead, like Ford and Honda. Only the build is limited: a release still goes out in the
# frame it is asked for, the ESP needs an atomic hand-back.
# The rate is scheduled on how deep the request is, not on whether there is a lead: engaging
# behind one, or dropping the set speed with one in sight, is the same comfort case as an empty
# road. The cruise clip can never ask past -1.2 m/s², so that whole band is comfort braking and
# builds slowly, while a request only the MPC or the model can produce — a lead braking hard, a
# cut-in — is let through at Ford's 3.5 m/s³ (Ford notes the stock system does 5).
# Holding the request back winds up LongControl's integrator: its error is the planner's aTarget
# against aEgo, so it keeps integrating what we refuse to deliver, and actuators.accel arrives
# deeper than the plan by roughly ki * 0.5 * (request / rate) — about 0.35 m/s² for a -1.2 m/s²
# request at the slow end with kiV 0.5, twice that at kiV 1.0. It decays as soon as the ramp
# catches up. The lower breakpoint is kept clear of that band so wind-up alone cannot unlock the
# fast rate; raise the slow end if the overshoot is felt on the road.
DECEL_BUILD_RATE_BP = [-2.5, -1.5]   # m/s², the request being built toward
DECEL_BUILD_RATE_V = [0.035, 0.010]  # m/s² per 100 Hz frame, 3.5 and 1.0 m/s³

# CMM (the engine ECU) takes a wheel-torque request, not an acceleration, so this map is
# the conversion the stock radar would have done. Calibrated against measured accel from
# the LongitudinalManeuverMode suite — see helper-scripts/accel_map.py, which imports
# these directly so its suggestions can never be against a stale copy.
# The -0.5 and 0.0 points were recalibrated from the ki integrator's steady-state offset
# during engaged regen braking (~170 N.m over-braking across the -0.5..0 region): holding
# 0 m/s² took ~170 N.m and delivering -0.5 m/s² took ~-130 N.m. Single-route evidence —
# re-check with accel_map.py after the next drive.
ACCEL_LOOKUP = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]     # m/s²
TORQUE_LOOKUP = [-400, -130, 170, 350, 645, 862, 1100]   # N.m


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
    self.hold = False
    self.launching = False
    self.drive_away_frames = 0
    self.accel_last = 0.0

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
    # longitudinal
    # starting = actuators.longControlState == LongCtrlState.starting and CS.out.vEgo <= self.CP.vEgoStarting
    # stopping = actuators.longControlState == LongCtrlState.stopping

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

    # rate-limit how fast a deceleration request may build, see DECEL_BUILD_RATE_BP
    if CC.longActive:
      build_rate = interp(actuators.accel, DECEL_BUILD_RATE_BP, DECEL_BUILD_RATE_V)
      self.accel_last = max(actuators.accel, self.accel_last - build_rate)
      accel = self.accel_last
    else:
      # keep the limiter primed with the car's own deceleration so the next engage builds
      # from what the car is doing, but never from a positive value, and leave the request
      # itself at the planner's zero so the gas-override torque path is unchanged
      self.accel_last = min(CS.out.aEgo, 0.0)
      accel = actuators.accel
    accel_cmd = accel + accel_slope

    brake_accel = -0.5

    # calculate Torque
    # 1100 N.m extrapolates the measured 434 N.m per m/s² slope to a true 2.0 m/s²; the
    # stock radar's observed ceiling is 986 N.m (sustained, no gas), so whether the CMM
    # accepts requests past ~1000 is unverified — check delivered accel with accel_map.py
    torque_nm = interp(accel_cmd, ACCEL_LOOKUP, TORQUE_LOOKUP)
    torque = max(-400, min(torque_nm, 1100))

    braking = accel_cmd < brake_accel and not CS.out.gasPressed
    if self.CP.openpilotLongitudinalControl:
      if CC.hudControl.leadVisible and sm is not None:
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

      # The PCM keeps cruiseState.enabled up for ~80 ms after the driver takes over, but
      # controlsd zeroes actuators.accel the moment longActive drops. Keying the emulation
      # off the PCM alone made those frames claim an active ACC with a *positive* torque
      # request while the car was still decelerating, and the ESP (UC_FREIN) latched
      # ACC_ETAT_DECEL_OR_ESP_STATUS = 3 (accFaulted) within 30 ms of the inversion.
      # Measured with the driver braking over a -1.25 m/s² command. Both signals have to
      # come off the same clock.
      # A gas press is an override, not a disengage: longActive drops but cruise stays on,
      # and the stock radar keeps ACC advertised as suspended (ACC_STATUS 5) instead of
      # flipping it off and back on. brakePressed keeps the inversion guard above.
      gas_override = CS.out.gasPressed and not CS.out.brakePressed
      long_enabled = CS.out.cruiseState.enabled and (CC.longActive or gas_override)

      decel_active = braking and long_enabled

      # Standstill hold / drive-away, mirroring the stock radar sequence (see DRIVE_AWAY_FRAMES).
      # Driven off the planner's own request rather than the pitch-compensated one: on a
      # descent the gravity term alone keeps accel_cmd negative, and the hold would never
      # release however hard the planner asked to move.
      # A gas press ends the hold immediately. The stock radar drops the decel request and
      # advertises ACC suspended when the driver takes over; holding the ESP instead put a
      # decel request up against the driver's own torque and latched the fault
      # (ACC_ETAT_DECEL_OR_ESP_STATUS 2 -> 0 -> 3 within ~0.5 s of the pedal).
      if not long_enabled or CS.out.gasPressed:
        self.hold = False
        self.launching = False
        self.drive_away_frames = 0
      elif self.hold:
        if actuators.accel > 0.0:
          self.hold = False
          self.drive_away_frames = DRIVE_AWAY_FRAMES
      elif self.drive_away_frames > 0:
        self.drive_away_frames -= 1
        if self.drive_away_frames == 0:
          self.launching = True
      elif self.launching:
        if CS.out.vEgo >= LAUNCH_COMPLETE_SPEED:
          self.launching = False
        elif CS.out.standstill and actuators.accel <= 0.0:  # lead stopped again before we got rolling
          self.launching = False
          self.hold = True
      elif CS.out.standstill and actuators.accel <= 0.0:
        self.hold = True
      drive_away = self.drive_away_frames > 0
      standstill_hold = self.hold or drive_away

      # the brakes are still on their way out through the launch, so ask for at least what
      # the stock radar used to break the hold, and advertise it as potential torque while
      # holding — the map's own value only takes over once it climbs past the floor
      if standstill_hold or self.launching:
        torque = max(torque, LAUNCH_TORQUE)

      # stand in for the radar for exactly as long as it is off the bus
      if self.radar_disabled and not self.radar_released and self.frame % 2 == 0:
        can_sends.append(create_HS2_DYN1_MDD_ETAT_2B6(self.packer, self.frame // 2, actuators.accel, decel_active, long_enabled,
                                                      CS.out.gasPressed, CS.out.brakePressed, CS.out.standstill, torque,
                                                      standstill_hold, self.launching))
        can_sends.append(create_HS2_DYN_MDD_ETAT_2F6(self.packer, decel_active or standstill_hold or self.launching,
                                                     CC.hudControl.leadVisible, self.bars, drive_away))

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
