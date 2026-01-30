import { useState, useEffect } from 'react';
import { trackingAPI } from '../services/api';
import './UnauthorizedVehicleTracker.css';

const UnauthorizedVehicleTracker = () => {
  const [trackingData, setTrackingData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showActiveOnly, setShowActiveOnly] = useState(true);

  useEffect(() => {
    fetchTrackingData();
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchTrackingData, 5000);
    return () => clearInterval(interval);
  }, [showActiveOnly]);

  const fetchTrackingData = async () => {
    try {
      const response = await trackingAPI.getAll(showActiveOnly);
      setTrackingData(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch tracking data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000); // difference in seconds

    if (diff < 60) return `${diff} seconds ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
    return date.toLocaleString();
  };

  const renderCameraPath = (cameras) => {
    return (
      <div className="camera-path">
        {cameras.map((camera, index) => (
          <span key={camera}>
            <span className="camera-icon">📹{camera.slice(-1)}</span>
            {index < cameras.length - 1 && <span className="camera-arrow">→</span>}
          </span>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-header">Unauthorized Vehicle Tracking</div>
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header flex-between">
        <span>Unauthorized Vehicle Tracking</span>
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={showActiveOnly}
            onChange={(e) => setShowActiveOnly(e.target.checked)}
          />
          <span>Show Active Only</span>
        </label>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {trackingData.length === 0 ? (
        <div className="no-data">
          No {showActiveOnly ? 'active ' : ''}unauthorized vehicles detected
        </div>
      ) : (
        <div className="table-container">
          <table className="table tracking-table">
            <thead>
              <tr>
                <th>Vehicle ID</th>
                <th>Plate Number</th>
                <th>Color</th>
                <th>Car Name</th>
                <th>Confidence</th>
                <th>Camera Path</th>
                <th>First Seen</th>
                <th>Last Seen</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {trackingData.map((record) => (
                <tr
                  key={record.id}
                  className={record.is_active ? 'active-vehicle' : 'inactive-vehicle'}
                >
                  <td className="vehicle-id">{record.tracking_id}</td>
                  <td className="plate-number">
                    {record.plate_number || <span className="no-plate">N/A</span>}
                  </td>
                  <td>{record.color || 'Unknown'}</td>
                  <td>{record.car_name || 'Unknown'}</td>
                  <td>
                    {record.confidence_score ? (
                      <span className="confidence-score">
                        {record.confidence_score.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="no-plate">N/A</span>
                    )}
                  </td>
                  <td>{renderCameraPath(record.cameras_passed)}</td>
                  <td>{formatTimestamp(record.first_seen_time)}</td>
                  <td>{formatTimestamp(record.last_seen_time)}</td>
                  <td>
                    <span className={`status-badge ${record.is_active ? 'active' : 'inactive'}`}>
                      {record.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="tracking-footer">
        <small>
          Auto-refreshing every 5 seconds | Total records: {trackingData.length}
        </small>
      </div>
    </div>
  );
};

export default UnauthorizedVehicleTracker;
