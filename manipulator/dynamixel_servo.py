import numpy as np


class Address:
    # [address, length]
    def __init__(self):
        # -----------EEPROM AREA----------------
        # feature
        self.ID = [7, 1]  # default:1, range:0 ~ 252
        self.BAUD_RATE = [8, 1]  # default:1, range:0 ~ 6, unit:1 [Mbps]
        self.Drive_Mode = [10, 1]  # default:0, range:0 ~ 13
        self.Operating_Mode = [11, 1]  # default:3, range:0 ~ 16, 0:current, 1:velocity, 3:position, 4:extended position, 5:current-based position, 16:PWM
        self.Protocol_Type = [13, 1]  # default:2, range:2 ~ 22
        self.Homing_Offset = [20, 4]  # default:0, range:-1,044,479 ~ 1,044,479, unit:1 [pulse]=0.088 [deg]

        # limitation
        self.Max_Voltage_Limit = [32, 2]  # default:140, range:55 ~ 140, unit:0.1 [V]
        self.Min_Voltage_Limit = [34, 2]  # default:55, range:55 ~ 140,, unit:0.1 [V]
        self.Current_Limit = [38, 2]  # default:910, range:0 ~ 910, unit:1 [mA]
        self.Velocity_Limit = [44, 4]  # default:320, range:0 ~ 2,047, unit:0.229 [rev/min]
        self.Max_Position_Limit = [48, 4]  # default:4,095, range:0 ~ 4,095, unit:1 [pulse]
        self.Min_Position_Limit = [52, 4]  # default:0, range:0 ~ 4,095, unit:1 [pulse]=0.088 [deg]

        # -----------RAM AREA----------------
        # the eeprom area can be set only when the torque is off
        self.TORQUE_ENABLE = [64, 1]  # default:0, range:0, 1
        self.LED = [65, 1]  # default:0, range:0, 1

        # for Velocity Control Mode
        self.Velocity_I_GAIN = [76, 2]  # default:1200, range:0 ~ 16,383
        self.Velocity_P_GAIN = [78, 2]  # default:40, range:0 ~ 16,383

        # position control
        self.Position_D_GAIN = [80, 2]  # default:0, range:0 ~ 16,383
        self.Position_I_GAIN = [82, 2]  # default:0, range:0 ~ 16,383
        self.Position_P_GAIN = [84, 2]  # default:900, range:0 ~ 16,383

        # feedforward
        self.Feedforward_2nd_Gain = [88, 2]  # default:0, range:0 ~ 16,383, description: Feedforward Acceleration Gain
        self.Feedforward_1st_Gain = [90, 2]  # default:0, range:0 ~ 16,383, description: Feedforward Velocity Gain

        # goals
        self.GOAL_CURRENT = [102, 2]  # unit:1.0 [mA]
        self.GOAL_VELOCITY = [104, 4]  # unit:0.229 [rev/min]
        self.GOAL_POSITION = [116, 4]  # unit:1 [pulse]=0.088 [deg]

        # present state
        self.PRESENT_POSITION = [132, 4]  # unit:1 [pulse]
        self.PRESENT_VELOCITY = [128, 4]  # unit:0.229 [rev/min]
        self.PRESENT_CURRENT = [126, 2]  # unit:1.0 [mA]
        self.PRESENT_POS_VEL = [128, 8]
        self.PRESENT_POS_VEL_CUR = [126, 10]   

        # -----------Scales----------------
        self.POSITION_Radians_Scale = 2.0 * np.pi / 4096
        self.POSITION_Degrees_Scale = 360.0 / 4096


class Scale:
    def __init__(self):
        self.POSITION_SCALE = 2.0 * np.pi / 4096  # 0.001534 rad (0.088 degrees)
        # See http://emanual.robotis.com/docs/en/dxl/x/xh430-v210/#goal-velocity
        self.VELOCITY_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
        self.CURRENT_SCALE = 1.34



class BaseServo:
    def __init__(self, servo_type: str):
        self.servo_type = servo_type
        self.protocol_version = 2.0
        self.command_success = 0
        self.address = Address()
        self.scale = Scale()



class XC330Servo(BaseServo):
    def __init__(self):
        super().__init__('xc330')



