import { useState } from 'react';
import { getRayfinClient } from './lib/rayfin-client';

type Screen = 'search' | 'results' | 'subscribe';

interface EarthquakeRow {
  ID: string;
  Title: string;
  'Place Description': string;
  Magnitude: number;
  'Significance Class': string;
  'Event Time': string;
  'Country Code': string;
}

interface KpiRow {
  name: string;
  value: string | number;
}

function formatKpi(name: string, value: string | number): string {
  const label = name.replace(/^\[|\]$/g, '').trim();
  if (typeof value !== 'number') return String(value);
  if (label.includes('%')) return (value * 100).toFixed(1) + '%';
  if (label.toLowerCase().includes('magnitude')) return value.toFixed(2);
  return value.toLocaleString();
}

function sigClass(cls: string) {
  if (cls === 'High') return '#ef4444';
  if (cls === 'Moderate') return '#f59e0b';
  return '#22c55e';
}

export default function App() {
  const [screen, setScreen] = useState<Screen>('search');
  const [countryCode, setCountryCode] = useState('');
  const [kpis, setKpis] = useState<KpiRow[]>([]);
  const [earthquakes, setEarthquakes] = useState<EarthquakeRow[]>([]);
  const [minMag, setMinMag] = useState(4.0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [subDone, setSubDone] = useState(false);

  async function handleSearch() {
    if (!countryCode.trim()) { setError('Enter a country code'); return; }
    setError('');
    setLoading(true);
    try {
      const client = await getRayfinClient();

      // KPI query
      const kpiResult = await client.connectors.sm_usgs_earthquake.executeQuery({
        query: `EVALUATE ROW("Total Earthquakes", CALCULATE([Total Earthquakes], 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}"), "Avg Magnitude", CALCULATE([Avg Magnitude], 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}"), "Max Magnitude", CALCULATE([Max Magnitude], 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}"), "Max Significance", CALCULATE([Max Significance], 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}"), "% High Significance", CALCULATE([% High Significance], 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}"))`,
      });
      const kpiRows: KpiRow[] =
        kpiResult.status === 'success' && kpiResult.table.rows.length > 0
          ? kpiResult.table.columns.map((col, i) => ({
              name: col.name,
              value: kpiResult.table.rows[0][i] as string | number,
            }))
          : [];

      // Earthquake list query
      const listResult = await client.connectors.sm_usgs_earthquake.executeQuery({
        query: `EVALUATE TOPN(50, FILTER('Earthquake Events', 'Earthquake Events'[Country Code] = "${countryCode.toUpperCase().trim()}" && 'Earthquake Events'[Event Time] >= DATE(${new Date(Date.now() - 30*24*60*60*1000).getFullYear()}, ${new Date(Date.now() - 30*24*60*60*1000).getMonth()+1}, ${new Date(Date.now() - 30*24*60*60*1000).getDate()})), 'Earthquake Events'[Event Time], DESC)`,
      });
      const eqRows: EarthquakeRow[] =
        listResult.status === 'success'
          ? listResult.table.rows.map(row => {
              const cols = listResult.table.columns.map(c => c.name);
              const obj: Record<string, unknown> = {};
              cols.forEach((col, i) => { obj[col] = row[i]; });
              return obj as unknown as EarthquakeRow;
            })
          : [];

      setKpis(kpiRows);
      setEarthquakes(eqRows);
      setScreen('results');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Query failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleSubscribe() {
    setLoading(true);
    try {
      const client = await getRayfinClient();
      const session = client.auth.getSession();
      await client.data.Subscription.create({
        countryCode: countryCode.toUpperCase().trim(),
        minMagnitude: minMag,
        createdAt: new Date(),
        userId: session.user?.id ?? '',
      });
      setSubDone(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Subscribe failed');
    } finally {
      setLoading(false);
    }
  }

  // Screen: Search
  if (screen === 'search') return (
    <div style={{ maxWidth: 600, margin: '60px auto', fontFamily: 'sans-serif', padding: '0 16px' }}>
      <h1 style={{ color: '#094780' }}>🌍 Earthquake Alert App</h1>
      <p style={{ color: '#555' }}>Enter a 2-letter country code to see recent seismic activity.</p>
      <input
        value={countryCode}
        onChange={e => setCountryCode(e.target.value.toUpperCase())}
        placeholder="e.g. US, JP, FI, TR"
        maxLength={2}
        style={{ fontSize: 20, padding: '8px 12px', width: '100%', boxSizing: 'border-box', marginBottom: 12, border: '1px solid #ccc', borderRadius: 6 }}
      />
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <button
        onClick={handleSearch}
        disabled={loading}
        style={{ background: '#094780', color: '#fff', border: 'none', borderRadius: 6, padding: '10px 28px', fontSize: 16, cursor: 'pointer', width: '100%' }}
      >
        {loading ? 'Loading…' : 'Search'}
      </button>

      {kpis.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 24 }}>
          {kpis.map(k => (
            <div key={k.name} style={{ flex: '1 1 140px', background: '#f3f4f6', borderRadius: 8, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', marginBottom: 4 }}>{k.name.replace(/^\[|\]$/g, '').trim()}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#094780' }}>{formatKpi(k.name, k.value)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // Screen: Results
  if (screen === 'results') return (
    <div style={{ maxWidth: 800, margin: '40px auto', fontFamily: 'sans-serif', padding: '0 16px' }}>
      <button onClick={() => setScreen('search')} style={{ background: 'none', border: 'none', color: '#094780', cursor: 'pointer', marginBottom: 12, fontSize: 14 }}>← Back</button>
      <h2 style={{ color: '#094780' }}>Recent Earthquakes — {countryCode.toUpperCase()}</h2>

      {kpis.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 20 }}>
          {kpis.map(k => (
            <div key={k.name} style={{ flex: '1 1 120px', background: '#f3f4f6', borderRadius: 8, padding: '10px 14px' }}>
              <div style={{ fontSize: 10, color: '#6b7280', textTransform: 'uppercase' }}>{k.name.replace(/^\[|\]$/g, '').trim()}</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#094780' }}>{formatKpi(k.name, k.value)}</div>
            </div>
          ))}
        </div>
      )}

      {earthquakes.length === 0
        ? <p style={{ color: '#555' }}>No earthquakes found in the last 30 days for {countryCode}.</p>
        : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <thead>
              <tr style={{ background: '#094780', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Time</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Location</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Mag</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Significance</th>
              </tr>
            </thead>
            <tbody>
              {earthquakes.map((eq, i) => (
                <tr key={eq.ID ?? i} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '7px 10px', color: '#374151' }}>{new Date(eq['Event Time']).toLocaleDateString()}</td>
                  <td style={{ padding: '7px 10px', color: '#374151' }}>{eq['Place Description']}</td>
                  <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: 700, color: '#094780' }}>{eq.Magnitude?.toFixed(1)}</td>
                  <td style={{ padding: '7px 10px' }}>
                    <span style={{ background: sigClass(eq['Significance Class']), color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>
                      {eq['Significance Class']}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }

      <button
        onClick={() => { setSubDone(false); setScreen('subscribe'); }}
        style={{ marginTop: 20, background: '#094780', color: '#fff', border: 'none', borderRadius: 6, padding: '10px 28px', fontSize: 15, cursor: 'pointer' }}
      >
        Subscribe to alerts for {countryCode}
      </button>
    </div>
  );

  // Screen: Subscribe
  return (
    <div style={{ maxWidth: 480, margin: '80px auto', fontFamily: 'sans-serif', padding: '0 16px' }}>
      <button onClick={() => setScreen('results')} style={{ background: 'none', border: 'none', color: '#094780', cursor: 'pointer', marginBottom: 12, fontSize: 14 }}>← Back</button>
      <h2 style={{ color: '#094780' }}>Subscribe to Alerts</h2>
      <p style={{ color: '#555' }}>Receive alerts for earthquakes in <strong>{countryCode}</strong> above your chosen magnitude.</p>

      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
        Minimum magnitude: {minMag.toFixed(1)}
      </label>
      <input
        type="range" min={1} max={9} step={0.1}
        value={minMag}
        onChange={e => setMinMag(parseFloat(e.target.value))}
        style={{ width: '100%', marginBottom: 20 }}
      />

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {subDone
        ? <p style={{ color: '#22c55e', fontWeight: 600 }}>✅ Subscribed! You will receive alerts for {countryCode} (M ≥ {minMag.toFixed(1)}).</p>
        : (
          <button
            onClick={handleSubscribe}
            disabled={loading}
            style={{ background: '#094780', color: '#fff', border: 'none', borderRadius: 6, padding: '10px 28px', fontSize: 16, cursor: 'pointer', width: '100%' }}
          >
            {loading ? 'Saving…' : 'Confirm Subscription'}
          </button>
        )
      }
    </div>
  );
}
