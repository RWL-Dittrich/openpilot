from opendbc.car.can_definitions import CanData


def psa_checksum(address: int, sig, d: bytearray) -> int:
  chk_ini = {0x452: 0x4, 0x38D: 0x7, 0x2f6: 0x8, 0x2b6: 0xC}.get(address, 0xB)
  byte = sig.start_bit // 8
  d[byte] &= 0x0F if sig.start_bit % 8 >= 4 else 0xF0
  checksum = sum((b >> 4) + (b & 0xF) for b in d)
  return (chk_ini - checksum) & 0xF


def create_lka_steering(packer, lat_active: bool, apply_angle: float, status: int, lka_drive_mode: int):
  # DRIVE 0 means normal D mode, 1 means B (brake) mode — echo raw value from car
  values = {
    'DRIVE': lka_drive_mode,
    'STATUS': status,
    'LXA_ACTIVATION': 1,
    'TORQUE_FACTOR': lat_active * 100,
    'SET_ANGLE': apply_angle,
  }

  return packer.make_can_msg('LANE_KEEP_ASSIST', 0, values)


def create_resume_acc(packer, counter, status, hs2_dat_mdd_cmd_452):
  hs2_dat_mdd_cmd_452['COUNTER'] = counter
  hs2_dat_mdd_cmd_452['COCKPIT_GO_ACC_REQUEST'] = status
  return packer.make_can_msg('HS2_DAT_MDD_CMD_452', 1, hs2_dat_mdd_cmd_452)


def create_drive_away_request(packer, hs2_dyn_mdd_etat_2f6):
  hs2_dyn_mdd_etat_2f6['DRIVE_AWAY_REQUEST'] = 0
  return packer.make_can_msg('HS2_DYN_MDD_ETAT_2F6', 1, hs2_dyn_mdd_etat_2f6)


# Radar, 50 Hz
def create_HS2_DYN1_MDD_ETAT_2B6(packer, frame: int, accel: float, decel_active: bool, enabled: bool,
                                 gasPressed: bool, brakePressed: bool, standstill: bool, torque: int,
                                 standstill_hold: bool = False, launching: bool = False):
  # TODO: check difference between GMP_POTENTIAL_WHEEL_TORQUE and GMP_WHEEL_TORQUE

  # Stock standstill sequence, measured across six stock-radar drive-aways from a stop
  # behind a lead:
  #   hold:   decel request active with the saturated -10.65 hold code, potential torque
  #           request 1 carrying a real positive torque (700-1000 N.m), no wheel torque
  #           request
  #   pulse:  DRIVE_AWAY_REQUEST goes up in 0x2F6 for ~40 ms while this frame stays in the
  #           hold pattern — sometimes skipped entirely
  #   launch: desired decel steps to the saturated *positive* end of the signal (+2.0) with
  #           the decel request still active and the wheel torque request up. That step is
  #           what the ESP releases on: it cleared its ARRET_VHL_ADAS hold 70-100 ms later
  #           in every stock case and rolled the brakes off over the next ~0.5 s. Sending
  #           the launch with a mid-range desired decel (+1.0) instead left the ESP holding
  #           through 1.4 s of rising wheel torque with the car not moving at all.
  #
  # While a deceleration request is active, ACC_STATUS must keep advertising active (4):
  # the stock radar holds the full active pattern after a driver brake press and then
  # drops every signal in one frame, and sending the off pattern (2) while the decel
  # request was still up latched the ESP fault within 50 ms. That includes the first
  # frames of a pedal press while still enabled: the pedal flag goes true a frame before
  # controlsd disengages, so the pedal-derived statuses (2/5) must wait for decel_active
  # to clear — the release has to be atomic, never partial.
  torque_mode = enabled and not decel_active and not standstill_hold
  decel_req = decel_active or standstill_hold or launching
  values = {
    'MDD_DESIRED_DECELERATION': -10.65 if standstill_hold else 2.0 if launching else accel if decel_active else 2.05, # m/s², 2.05 is the field's idle value
    'POTENTIAL_WHEEL_TORQUE_REQUEST': 2 if decel_active and not standstill_hold else (1 if enabled else 0),
    'MIN_TIME_FOR_DESIRED_GEAR': 6.2 if torque_mode or standstill_hold else 0.0,
    'GMP_POTENTIAL_WHEEL_TORQUE': torque if torque_mode or standstill_hold else -4000,
    'ACC_STATUS': 4 if decel_req else ((5 if gasPressed else 2 if brakePressed and not standstill else 4) if enabled else (2 if brakePressed else 3)),
    'GMP_WHEEL_TORQUE': torque if torque_mode else -4000,
    'WHEEL_TORQUE_REQUEST': 1 if torque_mode else 0, # TODO: test 1: high torque range 2: low torque range
    'AUTO_BRAKING_STATUS': 3, # AEB # TODO: testing ALWAYS ENABLED to resolve DTC errors if enabled else 3, # maybe disabled on too high steering angle
    'MDD_DECEL_TYPE': 1 if decel_req else 0,
    'MDD_DECEL_CONTROL_REQ': 1 if decel_req else 0,
  }

  return packer.make_can_msg('HS2_DYN1_MDD_ETAT_2B6', 1, values)


# Radar, 50 Hz
def create_HS2_DYN_MDD_ETAT_2F6(packer, decel_active: bool, lead_visible: bool, lead_distance_bars: int,
                                drive_away: bool = False):
  values = {
    'DRIVE_AWAY_REQUEST': drive_away,
    'TARGET_DETECTED': lead_visible,
    # 'REQUEST_TAKEOVER': 0, # TODO potential signal for HUD message from OP
    # 'BLIND_SENSOR': 0,
    # 'REQ_VISUAL_COLL_ALERT_ARC': 0,
    # 'REQ_AUDIO_COLL_ALERT_ARC': 0,
    # 'REQ_HAPTIC_COLL_ALERT_ARC': 0,
    # 'INTER_VEHICLE_DISTANCE': 255.5,#255.5, # TODO: <distance> if enabled else 255.5,
    # 'ARC_STATUS': 6,  # 12 after 50 frames (1 sec) after AUTO_BRAKING_STATUS else 6
    # 'AUTO_BRAKING_IN_PROGRESS': 0,
    # 'AEB_ENABLED': 0,
    'DISPLAY_INTERVEHICLE_TIME': 5.0, # TODO: <time to vehicle> if enabled else 6.2,
    'MDD_DECEL_CONTROL_REQ': decel_active,
    # 'AUTO_BRAKING_STATUS': 3, # AEB # TODO: testing ALWAYS ENABLED to resolve DTC errors if enabled else 3, # maybe disabled on too high steering angle
    'TARGET_POSITION': lead_distance_bars, # distance to lead car, far - 4, 3, 2, 1 - near
  }

  return packer.make_can_msg('HS2_DYN_MDD_ETAT_2F6', 1, values)


# TODO: do this in interface.py init()
# Disable radar ECU by setting it to programming mode
def create_disable_radar():
  addr = 0x6B6
  bus = 1
  dat = [0x02, 0x10, 0x02, 0x80]
  dat.extend([0x0] * (8 - len(dat)))

  return CanData(addr, bytes(dat), bus)


# Put the radar ECU back into the default session so it resumes broadcasting.
# Without this it stays in the programming session until the S3 timer expires,
# measured at 5.22 s after the last 0x6B6 frame, and the ESP flags ACC data
# invalid for the whole of that gap.
def create_enable_radar():
  addr = 0x6B6
  bus = 1
  dat = [0x02, 0x10, 0x01, 0x80]
  dat.extend([0x0] * (8 - len(dat)))

  return CanData(addr, bytes(dat), bus)
