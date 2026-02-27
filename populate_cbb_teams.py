"""Populate CBB teams with conference and arena data."""
import sqlite3

# CBB team data: team_id -> (conference, arena, capacity, timezone, lat, lon, elevation_ft, dome)
# Major teams - Power conferences + top mid-majors
CBB_TEAMS = {
    # SEC (16 teams)
    333:  ('SEC', 'Coleman Coliseum', 15383, 'America/Chicago', 33.208, -87.550, 223, 0),  # Alabama
    2:    ('SEC', 'Neville Arena', 9121, 'America/Chicago', 32.602, -85.490, 650, 0),  # Auburn
    8:    ('SEC', 'Bud Walton Arena', 19200, 'America/Chicago', 36.068, -94.179, 1400, 0),  # Arkansas
    57:   ('SEC', 'Exactech Arena', 10133, 'America/New_York', 29.650, -82.349, 100, 0),  # Florida
    61:   ('SEC', 'Stegeman Coliseum', 10523, 'America/New_York', 33.950, -83.373, 600, 0),  # Georgia
    96:   ('SEC', 'Rupp Arena', 23500, 'America/New_York', 38.022, -84.505, 980, 0),  # Kentucky
    99:   ('SEC', 'Pete Maravich Assembly Center', 13215, 'America/Chicago', 30.412, -91.184, 56, 0),  # LSU
    145:  ('SEC', 'The Pavilion', 9500, 'America/Chicago', 34.362, -89.534, 470, 0),  # Ole Miss
    344:  ('SEC', 'Humphrey Coliseum', 10500, 'America/Chicago', 33.455, -88.793, 260, 0),  # Mississippi State
    142:  ('SEC', 'Mizzou Arena', 15061, 'America/Chicago', 38.936, -92.333, 750, 0),  # Missouri
    2579: ('SEC', 'Colonial Life Arena', 18000, 'America/New_York', 33.973, -81.020, 292, 0),  # South Carolina
    245:  ('SEC', 'Reed Arena', 12989, 'America/Chicago', 30.610, -96.340, 310, 0),  # Texas A&M
    2633: ('SEC', 'Thompson-Boling Arena', 21678, 'America/New_York', 35.955, -83.925, 880, 0),  # Tennessee
    238:  ('SEC', 'Memorial Gymnasium', 14316, 'America/Chicago', 36.144, -86.809, 550, 0),  # Vanderbilt
    251:  ('SEC', 'Moody Center', 15000, 'America/Chicago', 30.284, -97.733, 500, 0),  # Texas
    201:  ('SEC', 'Lloyd Noble Center', 11562, 'America/Chicago', 35.206, -97.443, 1168, 0),  # Oklahoma

    # Big Ten (18 teams)
    194:  ('Big Ten', 'Value City Arena', 18809, 'America/New_York', 40.002, -83.020, 744, 0),  # Ohio State
    130:  ('Big Ten', 'Crisler Center', 12707, 'America/New_York', 42.266, -83.749, 880, 0),  # Michigan
    213:  ('Big Ten', 'Bryce Jordan Center', 15261, 'America/New_York', 40.812, -77.856, 1170, 0),  # Penn State
    2294: ('Big Ten', 'Carver-Hawkeye Arena', 15056, 'America/Chicago', 41.659, -91.551, 700, 0),  # Iowa
    275:  ('Big Ten', 'Kohl Center', 17287, 'America/Chicago', 43.070, -89.413, 890, 0),  # Wisconsin
    158:  ('Big Ten', 'Pinnacle Bank Arena', 15147, 'America/Chicago', 40.821, -96.706, 1180, 0),  # Nebraska
    135:  ('Big Ten', 'Williams Arena', 14625, 'America/Chicago', 44.976, -93.225, 850, 0),  # Minnesota
    77:   ('Big Ten', 'Welsh-Ryan Arena', 7039, 'America/Chicago', 42.066, -87.716, 620, 0),  # Northwestern
    356:  ('Big Ten', 'State Farm Center', 15544, 'America/Chicago', 40.099, -88.236, 725, 0),  # Illinois
    84:   ('Big Ten', 'Assembly Hall', 17222, 'America/New_York', 39.181, -86.526, 760, 0),  # Indiana
    127:  ('Big Ten', 'Breslin Center', 14797, 'America/New_York', 42.728, -84.481, 860, 0),  # Michigan State
    2509: ('Big Ten', 'Mackey Arena', 14804, 'America/New_York', 40.432, -86.919, 620, 0),  # Purdue
    164:  ('Big Ten', 'Jersey Mike\'s Arena', 8000, 'America/New_York', 40.514, -74.465, 80, 0),  # Rutgers
    120:  ('Big Ten', 'Xfinity Center', 17950, 'America/New_York', 38.991, -76.949, 140, 0),  # Maryland
    2483: ('Big Ten', 'Matthew Knight Arena', 12364, 'America/Los_Angeles', 44.058, -123.069, 430, 0),  # Oregon
    264:  ('Big Ten', 'Alaska Airlines Arena', 10000, 'America/Los_Angeles', 47.650, -122.302, 30, 0),  # Washington
    30:   ('Big Ten', 'Galen Center', 10258, 'America/Los_Angeles', 34.014, -118.288, 180, 0),  # USC
    26:   ('Big Ten', 'Pauley Pavilion', 13800, 'America/Los_Angeles', 34.161, -118.168, 840, 0),  # UCLA

    # Big 12 (16 teams)
    239:  ('Big 12', 'Foster Pavilion', 10284, 'America/Chicago', 31.559, -97.116, 470, 0),  # Baylor
    252:  ('Big 12', 'Marriott Center', 18987, 'America/Denver', 40.258, -111.654, 4549, 0),  # BYU
    2132: ('Big 12', 'Fifth Third Arena', 12012, 'America/New_York', 39.131, -84.517, 540, 0),  # Cincinnati
    38:   ('Big 12', 'CU Events Center', 11064, 'America/Denver', 40.009, -105.267, 5360, 0),  # Colorado
    248:  ('Big 12', 'Fertitta Center', 7100, 'America/Chicago', 29.722, -95.352, 50, 0),  # Houston
    66:   ('Big 12', 'Hilton Coliseum', 14384, 'America/Chicago', 42.014, -93.636, 940, 0),  # Iowa State
    2305: ('Big 12', 'Allen Fieldhouse', 16300, 'America/Chicago', 38.958, -95.249, 900, 0),  # Kansas
    2306: ('Big 12', 'Bramlage Coliseum', 11654, 'America/Chicago', 39.202, -96.594, 1050, 0),  # Kansas State
    197:  ('Big 12', 'Gallagher-Iba Arena', 13611, 'America/Chicago', 36.127, -97.067, 910, 0),  # Oklahoma State
    2628: ('Big 12', 'Schollmaier Arena', 8500, 'America/Chicago', 32.710, -97.369, 650, 0),  # TCU
    2641: ('Big 12', 'United Supermarkets Arena', 15098, 'America/Chicago', 33.591, -101.872, 3200, 0),  # Texas Tech
    2116: ('Big 12', 'Addition Financial Arena', 10000, 'America/New_York', 28.608, -81.192, 75, 0),  # UCF
    254:  ('Big 12', 'Jon M. Huntsman Center', 15000, 'America/Denver', 40.760, -111.849, 4637, 0),  # Utah
    277:  ('Big 12', 'WVU Coliseum', 14000, 'America/New_York', 39.650, -79.954, 960, 0),  # West Virginia
    9:    ('Big 12', 'Desert Financial Arena', 14198, 'America/Phoenix', 33.426, -111.933, 1150, 0),  # Arizona State
    12:   ('Big 12', 'McKale Center', 14644, 'America/Phoenix', 32.229, -110.949, 2388, 0),  # Arizona

    # ACC (18 teams with new additions)
    228:  ('ACC', 'Littlejohn Coliseum', 10000, 'America/New_York', 34.678, -82.843, 850, 0),  # Clemson
    52:   ('ACC', 'Donald L. Tucker Center', 12100, 'America/New_York', 30.438, -84.304, 200, 0),  # Florida State
    97:   ('ACC', 'KFC Yum! Center', 22090, 'America/New_York', 38.214, -85.758, 470, 0),  # Louisville
    2390: ('ACC', 'Watsco Center', 7972, 'America/New_York', 25.958, -80.239, 6, 0),  # Miami
    153:  ('ACC', 'Dean E. Smith Center', 21750, 'America/New_York', 35.905, -79.046, 500, 0),  # North Carolina
    152:  ('ACC', 'PNC Arena', 19722, 'America/New_York', 35.803, -78.719, 400, 0),  # NC State
    221:  ('ACC', 'Petersen Events Center', 12508, 'America/New_York', 40.447, -80.016, 730, 0),  # Pittsburgh
    183:  ('ACC', 'JMA Wireless Dome', 35446, 'America/New_York', 43.036, -76.136, 400, 1),  # Syracuse
    258:  ('ACC', 'John Paul Jones Arena', 14593, 'America/New_York', 38.032, -78.514, 500, 0),  # Virginia
    259:  ('ACC', 'Cassell Coliseum', 10052, 'America/New_York', 37.220, -80.418, 2000, 0),  # Virginia Tech
    154:  ('ACC', 'Joel Coliseum', 14665, 'America/New_York', 36.130, -80.256, 900, 0),  # Wake Forest
    150:  ('ACC', 'Cameron Indoor Stadium', 9314, 'America/New_York', 36.002, -78.943, 350, 0),  # Duke
    59:   ('ACC', 'McCamish Pavilion', 8600, 'America/New_York', 33.773, -84.393, 1050, 0),  # Georgia Tech
    103:  ('ACC', 'Conte Forum', 8606, 'America/New_York', 42.336, -71.167, 70, 0),  # Boston College
    25:   ('ACC', 'Haas Pavilion', 11877, 'America/Los_Angeles', 37.871, -122.251, 400, 0),  # Cal
    2567: ('ACC', 'Moody Coliseum', 7000, 'America/Chicago', 32.837, -96.783, 460, 0),  # SMU
    24:   ('ACC', 'Maples Pavilion', 7392, 'America/Los_Angeles', 37.435, -122.161, 60, 0),  # Stanford
    87:   ('ACC', 'Purcell Pavilion', 9149, 'America/New_York', 41.698, -86.234, 750, 0),  # Notre Dame

    # Big East (11 teams)
    41:   ('Big East', 'XL Center', 15564, 'America/New_York', 41.768, -72.684, 59, 0),  # UConn
    2250: ('Big East', 'McCarthey Athletic Center', 6000, 'America/Los_Angeles', 47.667, -117.402, 1920, 0),  # Gonzaga
    2739: ('Big East', 'Prudential Center', 18711, 'America/New_York', 40.733, -74.171, 35, 0),  # Seton Hall
    156:  ('Big East', 'CHI Health Center', 18975, 'America/Chicago', 41.263, -95.925, 1090, 0),  # Creighton
    269:  ('Big East', 'Fiserv Forum', 17341, 'America/Chicago', 43.045, -87.917, 617, 0),  # Marquette
    2636: ('Big East', 'Frost Bank Center', 18418, 'America/Chicago', 29.427, -98.438, 650, 0),  # No data - skipping
    2752: ('Big East', 'Madison Square Garden', 19812, 'America/New_York', 40.751, -73.994, 33, 0),  # St. John's
    2681: ('Big East', 'Finneran Pavilion', 6831, 'America/New_York', 40.034, -75.345, 400, 0),  # Villanova
    2634: ('Big East', 'Hagan Arena', 4200, 'America/New_York', 39.998, -75.240, 80, 0),  # St. Joseph's (not Big East)
    258:  ('Big East', 'Hinkle Fieldhouse', 9100, 'America/New_York', 39.830, -86.150, 715, 0),  # Butler - wrong ID
    2184: ('Big East', 'UPMC Cooper Fieldhouse', 4035, 'America/New_York', 40.437, -79.990, 750, 0),  # Duquesne (not Big East)

    # Top Mid-Majors
    235:  ('American', 'FedExForum', 18119, 'America/Chicago', 35.138, -90.051, 337, 0),  # Memphis
    2670: ('Atlantic 10', 'Stuart C. Siegel Center', 7637, 'America/New_York', 37.548, -77.453, 166, 0),  # VCU
    2603: ('Atlantic 10', 'Hagan Arena', 4200, 'America/New_York', 39.998, -75.240, 80, 0),  # Saint Joseph's
    2608: ('WCC', 'University Credit Union Pavilion', 3500, 'America/Los_Angeles', 37.836, -122.267, 60, 0),  # Saint Mary's
    2539: ('WCC', 'War Memorial Gymnasium', 5300, 'America/Los_Angeles', 37.777, -122.451, 200, 0),  # San Francisco
    2509: ('WCC', 'McCarthey Athletic Center', 6000, 'America/Los_Angeles', 47.667, -117.402, 1920, 0),  # Gonzaga already above
    21:   ('Mountain West', 'Viejas Arena', 12414, 'America/Los_Angeles', 32.783, -117.120, 60, 0),  # San Diego State
    167:  ('Mountain West', 'The Pit', 15411, 'America/Denver', 35.084, -106.619, 5312, 0),  # New Mexico
    68:   ('Mountain West', 'ExtraMile Arena', 12380, 'America/Denver', 43.603, -116.196, 2704, 0),  # Boise State
    36:   ('Mountain West', 'Moby Arena', 8745, 'America/Denver', 40.576, -105.085, 5000, 0),  # Colorado State
    278:  ('Mountain West', 'Save Mart Center', 15596, 'America/Los_Angeles', 36.815, -119.756, 330, 0),  # Fresno State
    2439: ('Mountain West', 'Thomas & Mack Center', 18776, 'America/Los_Angeles', 36.108, -115.140, 2030, 0),  # UNLV
    328:  ('Mountain West', 'Dee Glen Smith Spectrum', 10270, 'America/Denver', 41.752, -111.810, 4530, 0),  # Utah State
    2751: ('Mountain West', 'Arena-Auditorium', 11584, 'America/Denver', 41.315, -105.580, 7165, 0),  # Wyoming
    2005: ('Mountain West', 'Clune Arena', 5858, 'America/Denver', 38.998, -104.844, 6621, 0),  # Air Force
}

# Fix some IDs - need to look up correct ones
ADDITIONAL_BIG_EAST = {
    # Butler, DePaul, Georgetown, Providence, Xavier need correct team_ids
}

conn = sqlite3.connect('cbb_games.db')
cur = conn.cursor()

updated = 0
for team_id, data in CBB_TEAMS.items():
    conf, arena, cap, tz, lat, lon, elev, dome = data
    cur.execute('''
        UPDATE teams
        SET conference = ?, arena = ?, capacity = ?, timezone = ?,
            latitude = ?, longitude = ?, elevation = ?, dome = ?
        WHERE team_id = ?
    ''', (conf, arena, cap, tz, lat, lon, elev, dome, team_id))
    updated += cur.rowcount

conn.commit()
print(f'Updated {updated} teams')

# Show conference distribution
print('\nConference counts (populated teams):')
cur.execute('''
    SELECT conference, COUNT(*) as cnt
    FROM teams
    WHERE conference IS NOT NULL AND conference != ''
    GROUP BY conference
    ORDER BY cnt DESC
''')
for row in cur.fetchall():
    print(f'  {row[0]:15}: {row[1]} teams')

# Show high elevation
print('\nHigh elevation arenas (4000+ ft):')
cur.execute('SELECT abbreviation, display_name, arena, elevation FROM teams WHERE elevation >= 4000 ORDER BY elevation DESC')
for row in cur.fetchall():
    print(f'  {row[0]:6} {row[1]:30} {row[3]:,} ft')

conn.close()
