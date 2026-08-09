#!/usr/bin/env python3
import math
import unittest
import numpy as np

from opendbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm
from opendbc.car.psa.carcontroller import get_safety_CP
from opendbc.car.psa.values import CarControllerParams
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, away_round, round_speed

LANE_KEEP_ASSIST = 0x3F2


def floor_angle(apply_angle, can_offset=0):
  # SET_ANGLE is a signed signal with a 0.1 deg factor and no offset, so it represents 0 exactly.
  # safety truncates float limits to CAN units, so the at-limit angle is floor(limit) + tolerance
  return (math.floor(apply_angle / 0.1) + can_offset) * 0.1


class TestPsaSafetyBase(common.CarSafetyTest, common.AngleSteeringSafetyTest):
  RELAY_MALFUNCTION_ADDRS = {0: (LANE_KEEP_ASSIST,)}
  FWD_BLACKLISTED_ADDRS = {2: [LANE_KEEP_ASSIST]}
  TX_MSGS = [[1010, 0], [1106, 1], [1718, 1], [1942, 1], [1270, 1], [694, 1], [758, 1]]

  MAIN_BUS = 0
  ADAS_BUS = 1
  CAM_BUS = 2

  STEER_ANGLE_MAX = 390
  DEG_TO_CAN = 10

  # PSA uses get_max_angle_delta_vm and get_max_angle_vm for real lateral accel and jerk limits
  # TODO: integrate this into AngleSteeringSafetyTest
  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None

  # Real time limits
  LATERAL_FREQUENCY = 100  # Hz

  cnt_angle_cmd = 0

  def _get_steer_cmd_angle_max(self, speed):
    return get_max_angle_vm(max(speed, 1), self.VM, CarControllerParams)

  def setUp(self):
    self.VM = VehicleModel(get_safety_CP())
    self.packer = CANPackerSafety("psa_aee2010_r3")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.psa, 0)
    self.safety.init_tests()

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True):
    values = {"SET_ANGLE": angle, "TORQUE_FACTOR": 100 if enabled else 0}
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("LANE_KEEP_ASSIST", self.MAIN_BUS, values)

  def _angle_meas_msg(self, angle: float):
    values = {"ANGLE": angle}
    return self.packer.make_can_msg_safety("STEERING_ALT", self.MAIN_BUS, values)

  def _pcm_status_msg(self, enable):
    values = {"RVV_ACC_ACTIVATION_REQ": enable}
    return self.packer.make_can_msg_safety("HS2_DAT_MDD_CMD_452", self.ADAS_BUS, values)

  def _speed_msg(self, speed):
    kph = speed * 3.6
    values = {
      "P263_VehV_VPsvValWhlFrtL": kph,
      "P264_VehV_VPsvValWhlFrtR": kph,
      "P265_VehV_VPsvValWhlBckL": kph,
      "P266_VehV_VPsvValWhlBckR": kph,
    }
    return self.packer.make_can_msg_safety("Dyn4_FRE", self.MAIN_BUS, values)

  def _vehicle_moving_msg(self, speed: float):
    values = {"VEHICLE_STANDSTILL": 0 if speed > self.STANDSTILL_THRESHOLD else 1}
    return self.packer.make_can_msg_safety("HS2_DYN_UCF_MDD_32D", self.ADAS_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"P013_MainBrake": brake}
    return self.packer.make_can_msg_safety("Dat_BSI", self.CAM_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"GAS_PEDAL": int(gas * 100)}
    return self.packer.make_can_msg_safety("DRIVER", self.CAM_BUS, values)

  def test_rx_hook(self):
    # cruise
    for _ in range(10):
      self.assertTrue(self._rx(self._pcm_status_msg(0)))
    msg = self._pcm_status_msg(0)
    # invalidate checksum
    msg[0].data[5] = 0x00
    self.assertFalse(self._rx(msg))
    msg = self._pcm_status_msg(0)
    # write to unused payload byte
    msg[0].data[6] = 0xAB
    self.assertTrue(self._rx(msg))

  def test_angle_cmd_when_enabled(self):
    # We properly test lateral acceleration and jerk below
    pass

  def _round_speed_can(self, speed):
    # match wheel speed signal rounding on CAN (0.01 km/h factor)
    return round_speed(away_round(speed * 3.6 / 0.01) * 0.01 / 3.6)

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = self._round_speed_can(max(speed, 1))
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed

        # at limit (safety tolerance adds 1)
        max_angle = floor_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), 1) * sign
        max_angle = np.clip(max_angle, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self.safety.set_desired_angle_last(round(max_angle * self.DEG_TO_CAN))

        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle, True)))

        # 1 unit above limit
        max_angle_raw = floor_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), 2) * sign
        max_angle = np.clip(max_angle_raw, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self._tx(self._angle_cmd_msg(max_angle, True))

        # at low speeds max angle is above the max steer angle, so adding 1 has no effect
        should_tx = abs(max_angle_raw) >= self.STEER_ANGLE_MAX
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(max_angle, True)))

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = self._round_speed_can(max(speed, 1))
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed
        self._tx(self._angle_cmd_msg(0, True))

        # Stay within limits (safety tolerance adds 1)
        # Up
        max_angle_delta = floor_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams), 1) * sign
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Don't change
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        # Inject too high rates
        # Up
        max_angle_delta = floor_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams), 2) * sign
        self.assertFalse(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Don't change
        self.safety.set_desired_angle_last(round(max_angle_delta * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))

        # Recover
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))


class TestPsaStockSafety(TestPsaSafetyBase):
  pass


if __name__ == "__main__":
    unittest.main()
