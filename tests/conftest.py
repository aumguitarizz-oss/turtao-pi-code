import pytest
from turtao.hardware.mocks import MockSerialLink, MockCamera
from turtao.state import AppState
from turtao.config import Settings, AppConfig
from pathlib import Path

@pytest.fixture
def mock_serial():
    return MockSerialLink()

@pytest.fixture
def mock_camera():
    return MockCamera()

@pytest.fixture
def app_state():
    return AppState()

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def app_config():
    return AppConfig()
