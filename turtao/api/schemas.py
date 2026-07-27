
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    uptime_s: int


class AlertResponse(BaseModel):
    threat: bool
    confidence: float


class BatteryResponse(BaseModel):
    voltage: float
    current_ma: int
    percent: float
    status: str


class EnvironmentResponse(BaseModel):
    temp_c: float
    humidity_pct: int
    pressure_hpa: float
    gas_mq2: int
    air_quality_mq135: int
    sound_level: float
    motion: bool
    orientation: "OrientationModel"
    tof_cm: list[int]


class OrientationModel(BaseModel):
    pitch: float
    roll: float
    yaw: float


class ModeRequest(BaseModel):
    mode: str


class ModeResponse(BaseModel):
    mode: str


class ControlRequest(BaseModel):
    speed: float | None = None
    nerf: bool | None = None
    safe_mode: bool | None = None
    pan: int | None = None
    tilt: int | None = None


class MoveRequest(BaseModel):
    ml: float
    mr: float


class LedRequest(BaseModel):
    mode: str


class OkResponse(BaseModel):
    ok: bool


class SettingsResponse(BaseModel):
    hostname: str
    ble_proximity_enabled: bool
    phone_registration: str
    tts_event_toggles: dict
    intercom_volume: float
    face_tolerance: float
    anti_spoof_enabled: bool
    speed: float
    safe_mode: bool
    auto_flashbang: bool
    stealth_mode: bool
    notifications: dict


class BLEDevice(BaseModel):
    id: str
    name: str
    rssi: int
    owner: bool


class BLERegisterRequest(BaseModel):
    mac: str


class FaceSummary(BaseModel):
    name: str
    thumb_url: str
    poses: int


class UnknownFace(BaseModel):
    id: str
    image_url: str
    first_seen: str
    cluster_count: int


class EnrollStatus(BaseModel):
    active: bool
    pose_index: int = 0
    poses_total: int = 5
    quality_issue: str | None = None


class EnrollStartRequest(BaseModel):
    name: str


class PromoteRequest(BaseModel):
    face_id: str
    name: str


class EventsResponse(BaseModel):
    id: str
    type: str
    message: str
    at: str


class OTAVersionResponse(BaseModel):
    version: str


class MapResponse(BaseModel):
    grid: list[list[int]]
    trail: list[list[int]]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
