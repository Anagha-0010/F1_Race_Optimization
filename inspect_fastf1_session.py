import fastf1

fastf1.Cache.enable_cache('cache')

# === Change these as needed ===
year = 2023
race = 'Belgian Grand Prix'
session_type = 'R'   # Q for Quali, FP1, FP2, etc.
driver_code = 'VER'

# === Load session ===
session = fastf1.get_session(year, race, session_type)
session.load()

# === Explore top-level attributes ===
print("\nAvailable session attributes:")
print(dir(session))

print("Drivers in this race:", [session.get_driver(n)['Abbreviation'] for n in session.drivers])

print("\nWeather columns:")
print(session.weather_data.columns)
print(session.weather_data.head())

print("\nTrack Status Flags:")
print(session.track_status.head())

print("\nRace Control Messages:")
print(session.race_control_messages.head())

print("\nFinal Results:")
print(session.results[['Abbreviation', 'Position', 'GridPosition', 'Time', 'Status']])
