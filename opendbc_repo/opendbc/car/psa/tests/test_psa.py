import unittest
from types import SimpleNamespace

import numpy as np

from opendbc.can import CANParser, CANPacker
from opendbc.car import DT_CTRL, structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.psa.carcontroller import (DECEL_BUILD_RATE_V, LAUNCH_TORQUE, RADAR_DISABLE_FRAME,
                                          RADAR_ENABLE_TIMEOUT_FRAMES)
from opendbc.car.psa.carstate import CLUSTER_SETPOINT_OFFSET, RADAR_TIMEOUT_FRAMES

DISABLE_RADAR = (0x6B6, b'\x02\x10\x02\x80\x00\x00\x00\x00')
ENABLE_RADAR = (0x6B6, b'\x02\x10\x01\x80\x00\x00\x00\x00')
RADAR_EMULATION = (0x2B6, 0x2F6)
PSA_ADAS_BUS = 1


TOGGLES = SimpleNamespace()


def make_car_interface(alpha_long: bool):
  CarInterface = interfaces["PSA_PEUGEOT_208"]
  fingerprint = {i: {} for i in range(8)}
  CP = CarInterface.get_params("PSA_PEUGEOT_208", fingerprint, [],
                               alpha_long=alpha_long, is_release=False, docs=False, starpilot_toggles=TOGGLES)
  FPCP = CarInterface.get_starpilot_params("PSA_PEUGEOT_208", fingerprint, [], CP, TOGGLES)
  return CarInterface(CP, FPCP)


class TestPsaRadarKnockout(unittest.TestCase):
  """The radar ECU knockout has to stay in step with what is actually on the ADAS bus.

  The ESP (UC_FREIN) marks its ACC fields invalid after ~150 ms without 0x2B6, which
  openpilot reports as accFaulted, so 0x2B6 must never be missing from the bus for
  longer than that and must never be sent by two ECUs at once.
  """

  def setUp(self):
    self.CI = make_car_interface(alpha_long=True)
    self.assertTrue(self.CI.CP.openpilotLongitudinalControl)
    self.CC = structs.CarControl().as_reader()
    self.now_nanos = 0

  def step(self, radar_alive: bool):
    self.CI.update([], TOGGLES)
    self.CI.CS.radar_alive = radar_alive
    _, can_sends = self.CI.apply(self.CC, self.now_nanos, TOGGLES)
    self.now_nanos += int(DT_CTRL * 1e9)
    return can_sends

  @staticmethod
  def addrs(can_sends):
    return [msg[0] for msg in can_sends]

  def test_no_knockout_before_safety_mode_is_certain(self):
    # pandad still has the panda in ELM327 mode early on, where the diagnostic
    # knockout is allowed through but the emulation that replaces it is not
    for _ in range(RADAR_DISABLE_FRAME):
      addrs = self.addrs(self.step(radar_alive=True))
      self.assertNotIn(0x6B6, addrs)
      for addr in RADAR_EMULATION:
        self.assertNotIn(addr, addrs)

  def test_knockout_sent_once_then_waits_for_radar_to_go_quiet(self):
    for _ in range(RADAR_DISABLE_FRAME):
      self.step(radar_alive=True)

    # the knockout goes out exactly once
    assert DISABLE_RADAR in [(m[0], m[1]) for m in self.step(radar_alive=True)]

    # the real radar is still transmitting, so we must not emulate yet
    for _ in range(50):
      addrs = self.addrs(self.step(radar_alive=True))
      self.assertNotIn(0x6B6, addrs)
      for addr in RADAR_EMULATION:
        self.assertNotIn(addr, addrs, "emulating while the real radar is still on the bus")

    # once it goes quiet, emulation takes over
    seen: set[int] = set()
    for _ in range(4):
      seen.update(self.addrs(self.step(radar_alive=False)))
    self.assertTrue(RADAR_EMULATION[0] in seen and RADAR_EMULATION[1] in seen)

  def test_no_knockout_without_openpilot_longitudinal(self):
    self.CI = make_car_interface(alpha_long=False)
    self.assertFalse(self.CI.CP.openpilotLongitudinalControl)
    for _ in range(RADAR_DISABLE_FRAME * 2):
      addrs = self.addrs(self.step(radar_alive=True))
      self.assertNotIn(0x6B6, addrs)
      for addr in RADAR_EMULATION:
        self.assertNotIn(addr, addrs)


class TestPsaRadarHandoverBudget(unittest.TestCase):
  """The knockout hands 0x2B6 from the radar to openpilot, and the gap is on a clock.

  TestPsaRadarKnockout injects radar_alive directly, so it says nothing about how long
  that flag takes to turn over. That latency is the gap: the emulation does not start
  until CarState has counted RADAR_TIMEOUT_FRAMES of missing 0x2B6. A drive that spent
  the whole ESP budget there faulted the car.
  """

  # UC_FREIN marks its ACC fields invalid this long after the last 0x2B6. Measured
  # three ways in the 2026-08-15 logs: 150 ms, 160 ms, and a 152 ms fault.
  ESP_TOLERANCE = 0.150
  # worst 0x2B6 inter-frame gap over 4932 steady-state frames across 4 routes; the
  # median is 20.2 ms, so anything below this reads a live radar as a dead one
  WORST_RADAR_JITTER = 0.0354

  def silence(self):
    """Longest ADAS bus silence the knockout can open up, worst case."""
    # the radar's last frame can land a full period before the knockout goes out
    radar_period = 0.020
    # CarState needs this many 10 ms frames of no 0x2B6 to call the radar dead
    detection = RADAR_TIMEOUT_FRAMES * DT_CTRL
    # and the emulation is gated to even frames, so it can wait one more
    return radar_period + detection + DT_CTRL

  def test_emulation_starts_before_the_esp_gives_up(self):
    self.assertLess(self.silence(), self.ESP_TOLERANCE,
                    "0x2B6 goes missing for longer than the ESP tolerates; the car faults with accFaulted as alpha long engages")

  def test_emulation_starts_with_margin(self):
    # 150 ms is where it faulted, not where it is safe. The jitter floor below puts a
    # hard lower bound on the detection half of the budget, so 1.5x is what is on offer.
    self.assertLess(self.silence() * 1.5, self.ESP_TOLERANCE,
                    "no useful margin on the ESP's 150 ms tolerance")

  def test_detection_is_slower_than_the_radar_jitter(self):
    # too short and a live radar reads as dead, putting two ECUs on 0x2B6 at once
    self.assertGreater(RADAR_TIMEOUT_FRAMES * DT_CTRL, self.WORST_RADAR_JITTER * 1.5,
                       "would call the radar dead on normal 0x2B6 jitter")


class TestPsaRadarRelease(unittest.TestCase):
  """Switching alpha long off has to close the gap between us leaving 0x2B6 and the radar returning.

  This runs inside the control loop, on purpose: pandad puts the panda in NO_OUTPUT the
  moment deviceState goes offroad, so anything sent from the shutdown path is rejected and
  the radar only comes back on its own S3 timeout, ~5 s later. Measured at 4.42 s of
  silence, with the ESP latched for the whole of the next drive.
  """

  def setUp(self):
    self.CI = make_car_interface(alpha_long=True)
    self.CC = structs.CarControl().as_reader()
    self.now_nanos = 0

  def step(self, radar_alive: bool):
    self.CI.update([], TOGGLES)
    self.CI.CS.radar_alive = radar_alive
    _, can_sends = self.CI.apply(self.CC, self.now_nanos, TOGGLES)
    self.now_nanos += int(DT_CTRL * 1e9)
    return can_sends

  def knock_out(self):
    """Run up to a steady state with the radar off the bus and openpilot emulating it."""
    for _ in range(RADAR_DISABLE_FRAME + 1):
      self.step(radar_alive=True)
    for _ in range(10):
      self.step(radar_alive=False)
    assert self.CI.CC.radar_disabled

  @staticmethod
  def addrs(can_sends):
    return [msg[0] for msg in can_sends]

  def test_asks_the_radar_back_and_covers_the_gap(self):
    self.knock_out()

    self.assertFalse(self.CI.release_ecus(), "claimed to be done before touching the radar")
    sent = [(m[0], m[1]) for m in self.step(radar_alive=False)]
    self.assertIn(ENABLE_RADAR, sent, "did not ask the radar back")

    # keep standing in for it until it is transmitting again, and only ask once
    seen: set[int] = set()
    for _ in range(50):
      msgs = self.step(radar_alive=False)
      self.assertNotIn(ENABLE_RADAR, [(m[0], m[1]) for m in msgs])
      seen.update(self.addrs(msgs))
      self.assertFalse(self.CI.release_ecus())
    self.assertTrue(RADAR_EMULATION[0] in seen and RADAR_EMULATION[1] in seen, "left the bus silent while the radar restarted")

  def test_stops_emulating_as_soon_as_the_real_radar_transmits(self):
    self.knock_out()
    self.CI.release_ecus()
    for _ in range(10):
      self.step(radar_alive=False)

    # two ECUs on 0x2B6 would collide, so get out of the way on the first live frame
    self.step(radar_alive=True)
    self.assertTrue(self.CI.release_ecus(), "did not notice the radar came back")
    for _ in range(20):
      addrs = self.addrs(self.step(radar_alive=True))
      self.assertNotIn(0x6B6, addrs)
      for addr in RADAR_EMULATION:
        self.assertNotIn(addr, addrs, "still emulating after the radar returned")

  def test_tester_present_stops_so_the_session_can_end(self):
    self.knock_out()
    self.CI.release_ecus()
    for _ in range(300):
      self.assertNotIn(0x6B6, self.addrs(self.step(radar_alive=False))[1:], "kept the radar in its programming session")

  def test_terminates_if_the_radar_never_comes_back(self):
    self.knock_out()
    self.CI.release_ecus()
    for _ in range(RADAR_ENABLE_TIMEOUT_FRAMES - 1):
      self.step(radar_alive=False)
    self.assertFalse(self.CI.release_ecus())
    self.step(radar_alive=False)
    self.assertTrue(self.CI.release_ecus(), "would hold the onroad cycle forever")

  def test_nothing_to_hand_back_before_the_knockout(self):
    # toggled off inside the RADAR_DISABLE_FRAME window, we never touched the radar
    self.step(radar_alive=True)
    self.CI.release_ecus()
    sent = [(m[0], m[1]) for m in self.step(radar_alive=True)]
    self.assertTrue(self.CI.release_ecus())
    self.assertNotIn(ENABLE_RADAR, sent)

  def test_knockout_never_fires_after_a_release(self):
    self.knock_out()
    self.CI.release_ecus()
    self.step(radar_alive=True)
    for _ in range(RADAR_DISABLE_FRAME * 2):
      sent = [(m[0], m[1]) for m in self.step(radar_alive=True)]
      self.assertNotIn(DISABLE_RADAR, sent, "knocked the radar out again after handing it back")

  def test_release_is_a_noop_without_openpilot_longitudinal(self):
    self.CI = make_car_interface(alpha_long=False)
    self.assertTrue(self.CI.release_ecus())
    for _ in range(RADAR_DISABLE_FRAME * 2):
      self.assertNotIn(0x6B6, self.addrs(self.step(radar_alive=True)))


class PsaEmulationTest(unittest.TestCase):
  """Harness for driving the radar emulation frame by frame and decoding what went out."""

  def setUp(self):
    self.CI = make_car_interface(alpha_long=True)
    self.now_nanos = 0
    self.parser = CANParser('psa_aee2010_r3', [('HS2_DYN1_MDD_ETAT_2B6', 0)], PSA_ADAS_BUS)
    self.parser_2f6 = CANParser('psa_aee2010_r3', [('HS2_DYN_MDD_ETAT_2F6', 0)], PSA_ADAS_BUS)

  def step(self, long_active: bool, accel: float, brake_pressed: bool = False, gas_pressed: bool = False, cruise_enabled: bool = True,
           standstill: bool = False, v_ego: float = 0.0, pitch: float = 0.0, lead_visible: bool = False, a_ego: float = 0.0):
    CC = structs.CarControl()
    CC.longActive = long_active
    CC.actuators.accel = accel
    CC.orientationNED = [0.0, pitch, 0.0]
    CC.hudControl.leadVisible = lead_visible

    self.CI.update([], TOGGLES)
    self.CI.CS.radar_alive = False
    self.CI.CS.out.cruiseState.enabled = cruise_enabled
    self.CI.CS.out.brakePressed = brake_pressed
    self.CI.CS.out.gasPressed = gas_pressed
    self.CI.CS.out.standstill = standstill
    self.CI.CS.out.vEgo = v_ego
    self.CI.CS.out.aEgo = a_ego
    _, can_sends = self.CI.apply(CC.as_reader(), self.now_nanos, TOGGLES)
    self.now_nanos += int(DT_CTRL * 1e9)
    return can_sends

  def radar_msg(self, can_sends):
    """Decode the emulated 0x2B6 out of a frame's sends, or None if it did not go out."""
    frames = [(m[0], m[1], m[2]) for m in can_sends if m[0] == 0x2B6]
    if not frames:
      return None
    self.parser.update([[self.now_nanos, frames]])
    return self.parser.vl['HS2_DYN1_MDD_ETAT_2B6']

  def knock_out(self):
    for _ in range(RADAR_DISABLE_FRAME + 1):
      self.CI.update([], TOGGLES)
      self.CI.CS.radar_alive = True
      self.CI.apply(structs.CarControl().as_reader(), self.now_nanos, TOGGLES)
      self.now_nanos += int(DT_CTRL * 1e9)
    for _ in range(10):
      self.step(long_active=True, accel=0.0)
    assert self.CI.CC.radar_disabled

  def emulated(self, **kwargs):
    """Run frames until the 50 Hz emulation actually emits, and return the decoded message."""
    for _ in range(4):
      msg = self.radar_msg(self.step(**kwargs))
      if msg is not None:
        return msg
    self.fail("emulation never transmitted 0x2B6")

  def emulated_pair(self, **kwargs):
    """Like emulated(), but returns the decoded (0x2B6, 0x2F6) pair from the same frame."""
    for _ in range(4):
      sends = self.step(**kwargs)
      b6 = self.radar_msg(sends)
      if b6 is not None:
        f6 = [(m[0], m[1], m[2]) for m in sends if m[0] == 0x2F6]
        self.parser_2f6.update([[self.now_nanos, f6]])
        return b6, self.parser_2f6.vl['HS2_DYN_MDD_ETAT_2F6']
    self.fail("emulation never transmitted 0x2B6")

  def settled(self, **kwargs):
    """Hold a request until the decel rate limiter has caught up with it, see DECEL_BUILD_RATE_CRUISE."""
    for _ in range(int(4.0 / DT_CTRL)):
      self.step(**kwargs)
    return self.emulated(**kwargs)


class TestPsaBrakeOverride(PsaEmulationTest):
  """The emulated radar must never invert its request when the driver takes over.

  The PCM holds cruiseState.enabled up for ~80 ms after a driver brake press, but
  controlsd zeroes actuators.accel as soon as longActive drops. Driving the emulation
  off the PCM alone made those frames advertise an active ACC asking for *positive*
  wheel torque while the car was still decelerating, and the ESP (UC_FREIN) latched
  ACC_ETAT_DECEL_OR_ESP_STATUS = 3 within 30 ms.
  """

  def test_braking_frame_requests_deceleration(self):
    self.knock_out()
    msg = self.settled(long_active=True, accel=-1.25)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 1)
    self.assertAlmostEqual(msg['MDD_DESIRED_DECELERATION'], -1.25, places=1)
    self.assertEqual(int(msg['POTENTIAL_WHEEL_TORQUE_REQUEST']), 2, "not in brake mode")

  def test_driver_brake_never_inverts_into_a_torque_request(self):
    self.knock_out()
    self.settled(long_active=True, accel=-1.25)

    # driver brakes: longActive drops and accel is zeroed, but the PCM has not caught up yet
    msg = self.emulated(long_active=False, accel=0.0, brake_pressed=True, cruise_enabled=True)
    self.assertNotEqual(int(msg['POTENTIAL_WHEEL_TORQUE_REQUEST']), 1,
                        "asked the ESP for wheel torque while the driver was braking")
    self.assertEqual(msg['GMP_WHEEL_TORQUE'], -4000, "sent a torque value instead of the no-request sentinel")
    self.assertEqual(msg['GMP_POTENTIAL_WHEEL_TORQUE'], -4000)

  def test_release_is_atomic(self):
    # the stock radar releases an active deceleration by dropping every signal to idle in
    # the same frame — a partial release (the off/suspended status with the decel request
    # still up, or the request cleared with a stale desired decel) latched the ESP fault
    self.knock_out()
    self.settled(long_active=True, accel=-1.6)

    msg = self.emulated(long_active=False, accel=0.0, brake_pressed=True, cruise_enabled=True)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 0)
    self.assertAlmostEqual(msg['MDD_DESIRED_DECELERATION'], 2.05, places=1)
    self.assertEqual(int(msg['ACC_STATUS']), 2, "brake release must advertise ACC off with the request down")

  def test_gas_release_is_atomic(self):
    self.knock_out()
    self.settled(long_active=True, accel=-1.6)

    msg = self.emulated(long_active=True, accel=-1.6, gas_pressed=True)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 0)
    self.assertAlmostEqual(msg['MDD_DESIRED_DECELERATION'], 2.05, places=1)
    self.assertEqual(int(msg['ACC_STATUS']), 5, "gas over a decel suspends ACC, with the request down")

  def test_brake_press_while_still_enabled_keeps_acc_advertised_active(self):
    # brakePressed goes true a frame before controlsd drops longActive, so the off status
    # must wait for the decel request to clear: one frame of ACC_STATUS 2 with the decel
    # request still up latched the ESP fault
    self.knock_out()
    self.settled(long_active=True, accel=-1.25)

    msg = self.emulated(long_active=True, accel=-1.25, brake_pressed=True)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 1)
    self.assertEqual(int(msg['ACC_STATUS']), 4, "advertised ACC off while still requesting deceleration")

  def test_standstill_hold_and_drive_away(self):
    # the ESP holds the car at a stop for the radar and only releases after the stock
    # drive-away handshake: hold pattern (saturated -10.65 decel code, no wheel torque),
    # DRIVE_AWAY_REQUEST pulsed in 0x2F6, then the launch — desired decel at the saturated
    # positive end with the decel request still active, which is what the ESP lets go on
    self.knock_out()
    self.emulated(long_active=True, accel=-1.0, v_ego=1.0)

    # stopped: hold pattern, advertising the torque the powertrain is holding in reserve
    msg = self.emulated(long_active=True, accel=-0.2, standstill=True)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 1)
    self.assertAlmostEqual(msg['MDD_DESIRED_DECELERATION'], -10.65, places=1)
    self.assertEqual(int(msg['WHEEL_TORQUE_REQUEST']), 0)
    self.assertEqual(int(msg['ACC_STATUS']), 4)
    self.assertGreaterEqual(msg['GMP_POTENTIAL_WHEEL_TORQUE'], LAUNCH_TORQUE)

    # planner wants to move: the pulse goes up while 0x2B6 stays in the hold pattern
    b6, f6 = self.emulated_pair(long_active=True, accel=0.3, standstill=True)
    self.assertEqual(int(f6['DRIVE_AWAY_REQUEST']), 1)
    self.assertAlmostEqual(b6['MDD_DESIRED_DECELERATION'], -10.65, places=1)
    self.assertEqual(int(b6['WHEEL_TORQUE_REQUEST']), 0)

    # pulse over: launch, wheel torque up with the decel request still active
    for _ in range(100):
      b6, f6 = self.emulated_pair(long_active=True, accel=0.3, standstill=True)
    self.assertEqual(int(f6['DRIVE_AWAY_REQUEST']), 0)
    self.assertEqual(int(b6['WHEEL_TORQUE_REQUEST']), 1)
    self.assertGreaterEqual(b6['GMP_WHEEL_TORQUE'], LAUNCH_TORQUE, "launched under the torque the ESP releases on")
    self.assertEqual(int(b6['MDD_DECEL_CONTROL_REQ']), 1)
    self.assertAlmostEqual(b6['MDD_DESIRED_DECELERATION'], 2.0, places=1)

    # rolling: back to the plain torque pattern
    b6, f6 = self.emulated_pair(long_active=True, accel=0.3, v_ego=1.0)
    self.assertEqual(int(b6['MDD_DECEL_CONTROL_REQ']), 0)
    self.assertEqual(int(b6['WHEEL_TORQUE_REQUEST']), 1)

  def test_gas_press_ends_the_standstill_hold(self):
    # the driver pulling away by pedal is an override: keeping the hold up against it left
    # a decel request on the bus while the car crept forward and the ESP latched the fault
    self.knock_out()
    self.emulated(long_active=True, accel=-1.0, v_ego=1.0)
    msg = self.emulated(long_active=True, accel=-0.2, standstill=True)
    self.assertAlmostEqual(msg['MDD_DESIRED_DECELERATION'], -10.65, places=1)

    # gas press: longActive drops and accel is zeroed, cruise stays on
    msg = self.emulated(long_active=False, accel=0.0, gas_pressed=True, standstill=True)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 0, "held the ESP while the driver was on the pedal")
    self.assertNotAlmostEqual(msg['MDD_DESIRED_DECELERATION'], -10.65, places=1)
    self.assertEqual(int(msg['ACC_STATUS']), 5)

  def test_hold_releases_on_a_descent(self):
    # the hold is driven by the planner's request, not the pitch-compensated one: on a
    # descent gravity alone keeps the compensated value negative and the car would sit
    # there however hard the planner asked to move
    self.knock_out()
    self.emulated(long_active=True, accel=-1.0, v_ego=1.0, pitch=-0.1)
    self.emulated(long_active=True, accel=-0.2, standstill=True, pitch=-0.1)

    b6, f6 = self.emulated_pair(long_active=True, accel=0.3, standstill=True, pitch=-0.1)
    self.assertEqual(int(f6['DRIVE_AWAY_REQUEST']), 1, "hold never released on a downhill grade")

  def test_gas_override_suspends_acc_instead_of_turning_it_off(self):
    # a gas press drops longActive but cruise stays on; the stock radar advertises ACC
    # suspended (ACC_STATUS 5), not the off pattern, so the cluster must not flap
    self.knock_out()
    self.emulated(long_active=True, accel=0.5)

    msg = self.emulated(long_active=False, accel=0.0, gas_pressed=True, cruise_enabled=True)
    self.assertEqual(int(msg['ACC_STATUS']), 5, "advertised ACC off during a gas override")

  def test_gas_and_brake_together_never_request_torque(self):
    # both pedals: the brake wins, the gas override must not keep the enabled pattern up
    self.knock_out()
    self.settled(long_active=True, accel=-1.25)

    msg = self.emulated(long_active=False, accel=0.0, gas_pressed=True, brake_pressed=True, cruise_enabled=True)
    self.assertNotEqual(int(msg['POTENTIAL_WHEEL_TORQUE_REQUEST']), 1)
    self.assertEqual(msg['GMP_WHEEL_TORQUE'], -4000)
    self.assertNotEqual(int(msg['ACC_STATUS']), 5, "kept ACC suspended while the driver was braking")


class TestPsaDecelBuildRate(PsaEmulationTest):
  """A deceleration request has to build at a rate the driver can feel coming.

  The planner's cruise source steps. It is a P controller on speed error clipped at
  -1.2 m/s², so any set-speed change over 4.3 km/h saturates it, and its own jerk limiter
  runs while disengaged, leaving nothing to ramp: engaging above the set speed put -1.22
  on the bus in the first active frame and the car pulled -1.46 m/s² 0.8 s later. The car
  has no jerk signal to hand the ESP, so the shaping lives in the controller — see
  DECEL_BUILD_RATE_CRUISE.
  """

  def ramp(self, target: float, frames: int, **kwargs):
    """Hold a request for `frames` frames and return the limited value at each one."""
    out = []
    for _ in range(frames):
      self.step(long_active=True, accel=target, **kwargs)
      out.append(self.CI.CC.accel_last)
    return out

  def assert_build(self, target, **kwargs):
    """Ramp to `target` and return how long it took, checking the rate was never exceeded."""
    profile = self.ramp(target, 400, **kwargs)
    self.assertGreaterEqual(min(np.diff([0.0] + profile)), -max(DECEL_BUILD_RATE_V) - 1e-9,
                            "built faster than the fastest scheduled rate")
    self.assertAlmostEqual(profile[-1], target, msg="never reached the request")
    return next(i for i, a in enumerate(profile) if a <= target + 1e-9) * DT_CTRL

  def test_comfort_braking_builds_slowly(self):
    # everything the cruise clip can ask for lives in this band
    self.knock_out()
    self.assertGreater(self.assert_build(-1.2), 1.0, "the full request landed in under a second")

  def test_a_lead_in_sight_is_still_comfort_braking(self):
    # engaging behind a lead, or dropping the set speed with one in sight, asks for no more
    # than cruise tracking does, and must not be treated as urgent
    self.knock_out()
    self.assertGreater(self.assert_build(-1.2, lead_visible=True), 1.0, "a visible lead skipped the comfort rate")

  def test_a_hard_request_is_not_held_back(self):
    # a lead braking hard or a cut-in: only the MPC or the model can ask this deep
    self.knock_out()
    self.assertLess(self.assert_build(-3.0), 1.0, "held back a request that needed to go out now")

  def test_engaging_above_the_set_speed_does_not_step(self):
    self.knock_out()
    for _ in range(50):
      self.step(long_active=False, accel=0.0, v_ego=30.0)

    # first active frame with the planner already asking for the full cruise decel
    msg = self.emulated(long_active=True, accel=-1.2, v_ego=30.0)
    self.assertGreater(self.CI.CC.accel_last, -0.1, "landed the planner's saturated request in one frame")
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 0, "asked the ESP for the brakes on the engage frame")

  def test_the_cars_own_deceleration_is_not_re_ramped(self):
    # coasting down or coming off the brake, there is nothing to ease into: the limiter is
    # primed with aEgo while inactive so engaging picks up where the car already is
    self.knock_out()
    for _ in range(50):
      self.step(long_active=False, accel=0.0, v_ego=30.0, a_ego=-1.0)

    self.step(long_active=True, accel=-1.2, v_ego=30.0, a_ego=-1.0)
    self.assertLess(self.CI.CC.accel_last, -1.0, "made the driver wait through a ramp the car was already past")

  def test_release_and_acceleration_are_never_limited(self):
    # the ESP needs the hand-back in the frame it is asked for, and a rate limit on the way
    # out is what made braking feel like it came in steps — see the reverted DECEL_RELEASE_RATE
    self.knock_out()
    self.settled(long_active=True, accel=-1.2)

    msg = self.emulated(long_active=True, accel=0.0)
    self.assertEqual(self.CI.CC.accel_last, 0.0)
    self.assertEqual(int(msg['MDD_DECEL_CONTROL_REQ']), 0, "held a deceleration request after it was released")

    self.step(long_active=True, accel=1.0)
    self.assertEqual(self.CI.CC.accel_last, 1.0, "rate limited the accelerator")


class TestPsaClusterSetSpeed(unittest.TestCase):
  """openpilot's set speed has to read the same as the number on the dash.

  The bus carries a setpoint 2 km/h below the displayed one, which is what the car regulates
  to, so the display is corrected — see CLUSTER_SETPOINT_OFFSET. The control target is scaled
  by wheelSpeedFactor so we hold the same physical wheel speed stock ACC would.
  """

  def setUp(self):
    self.CI = make_car_interface(alpha_long=True)
    self.packer = CANPacker('psa_aee2010_r3')
    self.now_nanos = 0

  def setpoint(self, kph: int):
    """Put SPEED_SETPOINT on the ADAS bus and return the resulting CarState."""
    values = {'SPEED_SETPOINT': kph, 'RVV_ACC_ACTIVATION_REQ': 1}
    for _ in range(3):  # the parser wants a few frames before the values come through
      self.now_nanos += int(DT_CTRL * 1e9)
      msg = self.packer.make_can_msg('HS2_DAT_MDD_CMD_452', PSA_ADAS_BUS, values)
      CS, _ = self.CI.update([(self.now_nanos, [msg])], TOGGLES)
    return CS

  def test_cluster_set_speed_matches_the_dash(self):
    for kph in (50, 82, 130):
      CS = self.setpoint(kph)
      self.assertAlmostEqual(CS.cruiseState.speed * CV.MS_TO_KPH, kph * self.CI.CP.wheelSpeedFactor, places=3,
                             msg="control target no longer follows the value the car regulates to")
      self.assertAlmostEqual(CS.cruiseState.speedCluster * CV.MS_TO_KPH, kph + CLUSTER_SETPOINT_OFFSET, places=3,
                             msg="the displayed set speed is a step off the dash")

  def test_no_offset_with_acc_off(self):
    # the signal parks at 255 with ACC off, and 257 is not a set speed anyone should be shown
    CS = self.setpoint(255)
    self.assertEqual(CS.cruiseState.speedCluster, CS.cruiseState.speed)


if __name__ == "__main__":
  unittest.main()
