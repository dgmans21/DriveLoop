"""설정 dict → carla.WeatherParameters, 야간 가로등."""
from __future__ import annotations

import carla


def build_weather(params: dict[str, float]) -> carla.WeatherParameters:
    weather = carla.WeatherParameters()  # 새 객체 (프리셋 공유 객체를 수정하지 않도록)
    for key, value in params.items():
        if not hasattr(weather, key):
            raise ValueError(f"알 수 없는 날씨 파라미터: {key}")
        setattr(weather, key, float(value))
    return weather


def apply_weather(world: carla.World, params: dict[str, float]) -> None:
    weather = build_weather(params)
    world.set_weather(weather)
    # 해가 지면 가로등을 켠다
    lights = world.get_lightmanager()
    street = lights.get_all_lights(carla.LightGroup.Street)
    if weather.sun_altitude_angle < 0:
        lights.turn_on(street)
    else:
        lights.turn_off(street)


def weather_to_dict(weather: carla.WeatherParameters) -> dict[str, float]:
    keys = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
            "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance", "wetness")
    return {k: round(float(getattr(weather, k)), 3) for k in keys}
