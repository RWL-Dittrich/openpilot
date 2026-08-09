from dataclasses import dataclass, field

from opendbc.car.structs import CarParams
from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.lateral import AngleSteeringLimits, ISO_LATERAL_ACCEL
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, uds

Ecu = CarParams.Ecu

# Add extra tolerance for average banked road since safety doesn't have the roll.
# ~3.4 degrees, 6% superelevation. Higher actual roll lowers lateral acceleration.
AVERAGE_ROAD_ROLL = 0.06
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^2
# Lower than the ISO 11270 lateral jerk limit (5.0 m/s^3) with bank tolerance, matches safety MAX_LATERAL_JERK
MAX_LATERAL_JERK = 3.0 + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^3


class CarControllerParams:
  STEER_STEP = 1  # Angle command is sent at 100 Hz

  # v1 rate limits are unused: the PSA port limits via apply_steer_angle_limits_vm (v2).
  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    390,  # deg, EPS max
    ([0.], [0.]),
    ([0.], [0.]),
    MAX_LATERAL_ACCEL=MAX_LATERAL_ACCEL,
    MAX_LATERAL_JERK=MAX_LATERAL_JERK,
    # limit angle rate for low speed comfort, matches previous breakpoint-based rate limits at a standstill
    MAX_ANGLE_RATE=2,  # deg/10ms frame
  )
  STEER_DRIVER_ALLOWANCE = 5  # Driver intervention threshold, 0.5 Nm


@dataclass
class PSACarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.psa_a]))


@dataclass
class PSAPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'psa_aee2010_r3',
  })


class CAR(Platforms):
  PSA_PEUGEOT_208 = PSAPlatformConfig(
    [PSACarDocs("Peugeot 208 2019-25")],
    CarSpecs(mass=1530, wheelbase=2.54, steerRatio=18.3),
  )
  PSA_PEUGEOT_508 = PSAPlatformConfig(
    [PSACarDocs("Peugeot 508 2019-23")],
    CarSpecs(mass=1720, wheelbase=2.79, steerRatio=17.6),
  )


PSA_DIAG_REQ  = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0x01])
PSA_DIAG_RESP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0x01])

PSA_SERIAL_REQ = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER,  0xF1, 0x8C])
PSA_SERIAL_RESP = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40, 0xF1, 0x8C])

PSA_RX_OFFSET = -0x20

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [PSA_DIAG_REQ, PSA_SERIAL_REQ],
      [PSA_DIAG_RESP, PSA_SERIAL_RESP],
      rx_offset=PSA_RX_OFFSET,
      bus=0,
      obd_multiplexing=False,
    )
  ]
)

DBC = CAR.create_dbc_map()
