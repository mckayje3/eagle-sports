"""Populate CFB teams with stadium data."""
import sqlite3

# CFB stadium data: team_id -> (stadium, capacity, timezone, lat, lon, elevation_ft, dome)
CFB_TEAMS = {
    # SEC
    333:  ('Bryant-Denny Stadium', 100077, 'America/Chicago', 33.208, -87.550, 223, 0),  # Alabama
    2:    ('Jordan-Hare Stadium', 87451, 'America/Chicago', 32.602, -85.490, 650, 0),  # Auburn
    8:    ('Donald W. Reynolds Razorback Stadium', 76000, 'America/Chicago', 36.068, -94.179, 1400, 0),  # Arkansas
    57:   ('Ben Hill Griffin Stadium', 88548, 'America/New_York', 29.650, -82.349, 100, 0),  # Florida
    61:   ('Sanford Stadium', 92746, 'America/New_York', 33.950, -83.373, 600, 0),  # Georgia
    96:   ('Kroger Field', 61000, 'America/New_York', 38.022, -84.505, 980, 0),  # Kentucky
    99:   ('Tiger Stadium', 102321, 'America/Chicago', 30.412, -91.184, 56, 0),  # LSU
    145:  ('Vaught-Hemingway Stadium', 64038, 'America/Chicago', 34.362, -89.534, 470, 0),  # Ole Miss
    344:  ('Davis Wade Stadium', 61337, 'America/Chicago', 33.455, -88.793, 260, 0),  # Mississippi State
    142:  ('Faurot Field', 71168, 'America/Chicago', 38.936, -92.333, 750, 0),  # Missouri
    2579: ('Williams-Brice Stadium', 77559, 'America/New_York', 33.973, -81.020, 292, 0),  # South Carolina
    245:  ('Kyle Field', 102733, 'America/Chicago', 30.610, -96.340, 310, 0),  # Texas A&M
    2633: ('Neyland Stadium', 101915, 'America/New_York', 35.955, -83.925, 880, 0),  # Tennessee
    238:  ('FirstBank Stadium', 40350, 'America/Chicago', 36.144, -86.809, 550, 0),  # Vanderbilt
    251:  ('Darrell K Royal-Texas Memorial Stadium', 100119, 'America/Chicago', 30.284, -97.733, 500, 0),  # Texas
    201:  ('Gaylord Family Oklahoma Memorial Stadium', 80126, 'America/Chicago', 35.206, -97.443, 1168, 0),  # Oklahoma

    # Big Ten
    194:  ('Ohio Stadium', 102780, 'America/New_York', 40.002, -83.020, 744, 0),  # Ohio State
    130:  ('Michigan Stadium', 107601, 'America/New_York', 42.266, -83.749, 880, 0),  # Michigan
    213:  ('Beaver Stadium', 106572, 'America/New_York', 40.812, -77.856, 1170, 0),  # Penn State
    2294: ('Kinnick Stadium', 69250, 'America/Chicago', 41.659, -91.551, 700, 0),  # Iowa
    275:  ('Camp Randall Stadium', 80321, 'America/Chicago', 43.070, -89.413, 890, 0),  # Wisconsin
    158:  ('Memorial Stadium', 85458, 'America/Chicago', 40.821, -96.706, 1180, 0),  # Nebraska
    135:  ('Huntington Bank Stadium', 50805, 'America/Chicago', 44.976, -93.225, 850, 0),  # Minnesota
    77:   ('Ryan Field', 47130, 'America/Chicago', 42.066, -87.716, 620, 0),  # Northwestern
    356:  ('Memorial Stadium', 60670, 'America/Chicago', 40.099, -88.236, 725, 0),  # Illinois
    84:   ('Memorial Stadium', 52929, 'America/New_York', 39.181, -86.526, 760, 0),  # Indiana
    127:  ('Spartan Stadium', 75005, 'America/New_York', 42.728, -84.481, 860, 0),  # Michigan State
    2509: ('Ross-Ade Stadium', 57236, 'America/New_York', 40.432, -86.919, 620, 0),  # Purdue
    164:  ('SHI Stadium', 52454, 'America/New_York', 40.514, -74.465, 80, 0),  # Rutgers
    120:  ('SECU Stadium', 51802, 'America/New_York', 38.991, -76.949, 140, 0),  # Maryland
    2483: ('Autzen Stadium', 54000, 'America/Los_Angeles', 44.058, -123.069, 430, 0),  # Oregon
    264:  ('Husky Stadium', 70083, 'America/Los_Angeles', 47.650, -122.302, 30, 0),  # Washington
    30:   ('LA Memorial Coliseum', 77500, 'America/Los_Angeles', 34.014, -118.288, 180, 0),  # USC
    26:   ('Rose Bowl', 88565, 'America/Los_Angeles', 34.161, -118.168, 840, 0),  # UCLA

    # Big 12
    239:  ('McLane Stadium', 45140, 'America/Chicago', 31.559, -97.116, 470, 0),  # Baylor
    252:  ('LaVell Edwards Stadium', 63470, 'America/Denver', 40.258, -111.654, 4549, 0),  # BYU
    2132: ('Nippert Stadium', 40000, 'America/New_York', 39.131, -84.517, 540, 0),  # Cincinnati
    38:   ('Folsom Field', 50183, 'America/Denver', 40.009, -105.267, 5360, 0),  # Colorado
    248:  ('TDECU Stadium', 40000, 'America/Chicago', 29.722, -95.352, 50, 0),  # Houston
    66:   ('Jack Trice Stadium', 61500, 'America/Chicago', 42.014, -93.636, 940, 0),  # Iowa State
    2305: ('David Booth Kansas Memorial Stadium', 47233, 'America/Chicago', 38.958, -95.249, 900, 0),  # Kansas
    2306: ('Bill Snyder Family Stadium', 50000, 'America/Chicago', 39.202, -96.594, 1050, 0),  # Kansas State
    197:  ('Boone Pickens Stadium', 55509, 'America/Chicago', 36.127, -97.067, 910, 0),  # Oklahoma State
    2628: ('Amon G. Carter Stadium', 47000, 'America/Chicago', 32.710, -97.369, 650, 0),  # TCU
    2641: ('Jones AT&T Stadium', 60454, 'America/Chicago', 33.591, -101.872, 3200, 0),  # Texas Tech
    2116: ('FBC Mortgage Stadium', 45301, 'America/New_York', 28.608, -81.192, 75, 0),  # UCF
    254:  ('Rice-Eccles Stadium', 51444, 'America/Denver', 40.760, -111.849, 4637, 0),  # Utah
    277:  ('Mountaineer Field', 60000, 'America/New_York', 39.650, -79.954, 960, 0),  # West Virginia
    9:    ('Mountain America Stadium', 53599, 'America/Phoenix', 33.426, -111.933, 1150, 0),  # Arizona State
    12:   ('Arizona Stadium', 50782, 'America/Phoenix', 32.229, -110.949, 2388, 0),  # Arizona
    36:   ('Canvas Stadium', 41000, 'America/Denver', 40.576, -105.085, 5000, 0),  # Colorado State

    # ACC
    228:  ('Memorial Stadium', 81500, 'America/New_York', 34.678, -82.843, 850, 0),  # Clemson
    52:   ('Doak Campbell Stadium', 79560, 'America/New_York', 30.438, -84.304, 200, 0),  # Florida State
    97:   ('L&N Federal Credit Union Stadium', 42000, 'America/New_York', 38.214, -85.758, 470, 0),  # Louisville
    2390: ('Hard Rock Stadium', 64767, 'America/New_York', 25.958, -80.239, 6, 0),  # Miami
    153:  ('Kenan Memorial Stadium', 50500, 'America/New_York', 35.905, -79.046, 500, 0),  # North Carolina
    152:  ('Carter-Finley Stadium', 57583, 'America/New_York', 35.803, -78.719, 400, 0),  # NC State
    221:  ('Acrisure Stadium', 68400, 'America/New_York', 40.447, -80.016, 730, 0),  # Pittsburgh
    183:  ('JMA Wireless Dome', 49262, 'America/New_York', 43.036, -76.136, 400, 1),  # Syracuse (DOME)
    258:  ('Scott Stadium', 61500, 'America/New_York', 38.032, -78.514, 500, 0),  # Virginia
    259:  ('Lane Stadium', 66233, 'America/New_York', 37.220, -80.418, 2000, 0),  # Virginia Tech
    154:  ('Allegacy Federal Credit Union Stadium', 31500, 'America/New_York', 36.130, -80.256, 900, 0),  # Wake Forest
    150:  ('Wallace Wade Stadium', 40004, 'America/New_York', 36.002, -78.943, 350, 0),  # Duke
    59:   ('Bobby Dodd Stadium', 55000, 'America/New_York', 33.773, -84.393, 1050, 0),  # Georgia Tech
    103:  ('Alumni Stadium', 44500, 'America/New_York', 42.336, -71.167, 70, 0),  # Boston College
    25:   ('California Memorial Stadium', 62467, 'America/Los_Angeles', 37.871, -122.251, 400, 0),  # Cal
    2567: ('Gerald J. Ford Stadium', 32000, 'America/Chicago', 32.837, -96.783, 460, 0),  # SMU
    24:   ('Stanford Stadium', 50424, 'America/Los_Angeles', 37.435, -122.161, 60, 0),  # Stanford

    # Notre Dame (Independent)
    87:   ('Notre Dame Stadium', 77622, 'America/New_York', 41.698, -86.234, 750, 0),  # Notre Dame

    # Top Group of 5
    68:   ('Albertsons Stadium', 36387, 'America/Denver', 43.603, -116.196, 2704, 0),  # Boise State
    235:  ('Simmons Bank Liberty Stadium', 58325, 'America/Chicago', 35.121, -89.992, 280, 0),  # Memphis
    2655: ('Yulman Stadium', 30000, 'America/Chicago', 29.943, -90.118, 3, 0),  # Tulane
    2026: ('Kidd Brewer Stadium', 30000, 'America/New_York', 36.215, -81.685, 3333, 0),  # Appalachian State
    278:  ('Valley Childrens Stadium', 41031, 'America/Los_Angeles', 36.815, -119.756, 330, 0),  # Fresno State
    2005: ('Falcon Stadium', 46692, 'America/Denver', 38.998, -104.844, 6621, 0),  # Air Force
    2439: ('Allegiant Stadium', 65000, 'America/Los_Angeles', 36.091, -115.183, 2030, 1),  # UNLV (DOME)
    21:   ('Snapdragon Stadium', 35000, 'America/Los_Angeles', 32.783, -117.120, 60, 0),  # San Diego State
    349:  ('Michie Stadium', 38000, 'America/New_York', 41.390, -73.967, 180, 0),  # Army
    2426: ('Navy-Marine Corps Memorial Stadium', 34000, 'America/New_York', 38.991, -76.569, 50, 0),  # Navy
    204:  ('Reser Stadium', 35362, 'America/Los_Angeles', 44.559, -123.282, 235, 0),  # Oregon State
    265:  ('Gesa Field', 33000, 'America/Los_Angeles', 46.731, -117.168, 2350, 0),  # Washington State
}

conn = sqlite3.connect('cfb_games.db')
cur = conn.cursor()

updated = 0
for team_id, (stadium, cap, tz, lat, lon, elev, dome) in CFB_TEAMS.items():
    cur.execute('''
        UPDATE teams
        SET stadium = ?, capacity = ?, timezone = ?, latitude = ?, longitude = ?, elevation = ?, dome = ?
        WHERE team_id = ?
    ''', (stadium, cap, tz, lat, lon, elev, dome, team_id))
    updated += cur.rowcount

conn.commit()
print(f'Updated {updated} teams')

# Show dome stadiums
print()
print('Dome stadiums:')
cur.execute('SELECT abbreviation, display_name, stadium FROM teams WHERE dome = 1')
for row in cur.fetchall():
    print(f'  {row[0]:6} {row[1]:35} {row[2]}')

# Show high elevation
print()
print('High elevation stadiums (3000+ ft):')
cur.execute('SELECT abbreviation, display_name, elevation FROM teams WHERE elevation >= 3000 ORDER BY elevation DESC')
for row in cur.fetchall():
    print(f'  {row[0]:6} {row[1]:35} {row[2]:,} ft')

conn.close()
