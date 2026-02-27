import sqlite3

conn = sqlite3.connect('users.db')

CORRECTION = 3.5

q = '''
SELECT game_date, away_team, home_team,
       predicted_spread, vegas_spread
FROM prediction_cache
WHERE sport = 'CFB'
  AND vegas_spread IS NOT NULL
  AND predicted_spread IS NOT NULL
  AND game_date >= '2025-12-27'
ORDER BY game_date
'''

print('='*70)
print('CFB BOWL PICKS - WITH +3.5 CORRECTION')
print('='*70)

rows = conn.execute(q).fetchall()

games = []
for r in rows:
    date, away, home, model_spread, vegas_spread = r
    adj_spread = model_spread + CORRECTION
    adj_edge = adj_spread - vegas_spread

    games.append({
        'date': date[:10],
        'away': away,
        'home': home,
        'adj_spread': adj_spread,
        'vegas': vegas_spread,
        'adj_edge': adj_edge
    })

games.sort(key=lambda x: (x['date'], -abs(x['adj_edge'])))

current_date = None
for g in games:
    if g['date'] != current_date:
        current_date = g['date']
        print('')
        print(current_date + ':')
        print('-'*60)

    bet_side = g['away'] if g['adj_edge'] > 0 else g['home']
    marker = '>>>' if abs(g['adj_edge']) >= 2 else '   '

    line1 = marker + ' ' + g['away'][:18].ljust(18) + ' @ ' + g['home'][:18].ljust(18)
    line2 = '    Vegas: %+.1f | Adj Model: %+.1f | Edge: %+.1f' % (g['vegas'], g['adj_spread'], g['adj_edge'])
    print(line1)
    print(line2)
    if abs(g['adj_edge']) >= 2:
        print('    BET: ' + bet_side)

print('')
print('='*70)
print('BEST BETS (2+ pt edge after correction):')
print('='*70)
good_bets = [g for g in games if abs(g['adj_edge']) >= 2]
for g in sorted(good_bets, key=lambda x: -abs(x['adj_edge'])):
    bet_side = g['away'] if g['adj_edge'] > 0 else g['home']
    print('%s: %s (%+.1f edge)' % (g['date'], bet_side, g['adj_edge']))

conn.close()
