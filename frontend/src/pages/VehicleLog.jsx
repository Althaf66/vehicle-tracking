import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { trackingAPI } from '../services/api';
import './VehicleLog.css';

const CAMERA_INFO = [
  { id: 'camera1', name: 'Camera 1', label: 'Entrance', sublabel: 'LPR' },
  { id: 'camera2', name: 'Camera 2', label: 'EC', sublabel: '' },
  { id: 'camera3', name: 'Camera 3', label: 'Sports', sublabel: '' },
  { id: 'camera4', name: 'Camera 4', label: 'Exit', sublabel: '' },
];

const VehicleLog = () => {
  const [trackingData, setTrackingData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showActiveOnly, setShowActiveOnly] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [showActiveOnly]);

  const fetchData = async () => {
    try {
      const response = await trackingAPI.getAll(showActiveOnly);
      setTrackingData(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch vehicle log data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const mins = String(date.getMinutes()).padStart(2, '0');
    const secs = String(date.getSeconds()).padStart(2, '0');
    return `${day}/${month} ${hours}:${mins}:${secs}`;
  };

  const formatTimeOnly = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const hours = String(date.getHours()).padStart(2, '0');
    const mins = String(date.getMinutes()).padStart(2, '0');
    const secs = String(date.getSeconds()).padStart(2, '0');
    return `${hours}:${mins}:${secs}`;
  };

  const filteredData = trackingData.filter((record) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      (record.plate_number && record.plate_number.toLowerCase().includes(query)) ||
      (record.color && record.color.toLowerCase().includes(query)) ||
      (record.car_name && record.car_name.toLowerCase().includes(query)) ||
      (record.tracking_id && record.tracking_id.toLowerCase().includes(query))
    );
  });

  const renderCameraRoute = (cameras) => {
    return (
      <div className="camera-route-inline">
        {cameras.map((camera, index) => (
          <span key={camera}>
            <span className="route-badge">{camera.replace('camera', 'Cam ')}</span>
            {index < cameras.length - 1 && <span className="route-arrow">&rarr;</span>}
          </span>
        ))}
      </div>
    );
  };

  const renderRouteMap = () => {
    const passedCameras = selectedVehicle ? selectedVehicle.cameras_passed : [];
    const timestamps = selectedVehicle ? selectedVehicle.camera_timestamps : {};

    return (
      <div className="route-map-card">
        <div className="card-header">
          Camera Route Map
          {selectedVehicle && (
            <span className="route-map-vehicle">
              Vehicle: {selectedVehicle.tracking_id.substring(0, 8)}...
              {selectedVehicle.plate_number && ` | Plate: ${selectedVehicle.plate_number}`}
            </span>
          )}
        </div>
        <div className="route-map-container">
          <svg viewBox="0 0 800 160" className="route-map-svg">
            {/* Connection lines */}
            {CAMERA_INFO.map((cam, index) => {
              if (index >= CAMERA_INFO.length - 1) return null;
              const nextCam = CAMERA_INFO[index + 1];
              const isActive =
                passedCameras.includes(cam.id) && passedCameras.includes(nextCam.id);
              const x1 = 100 + index * 200 + 50;
              const x2 = 100 + (index + 1) * 200 - 50;
              return (
                <g key={`line-${index}`}>
                  <line
                    x1={x1}
                    y1={55}
                    x2={x2}
                    y2={55}
                    className={`route-line ${isActive ? 'active' : ''}`}
                  />
                  {/* Arrow head */}
                  <polygon
                    points={`${x2 - 8},${50} ${x2},${55} ${x2 - 8},${60}`}
                    className={`route-arrow-head ${isActive ? 'active' : ''}`}
                  />
                </g>
              );
            })}

            {/* Camera nodes */}
            {CAMERA_INFO.map((cam, index) => {
              const cx = 100 + index * 200;
              const isActive = passedCameras.includes(cam.id);
              const camTimestamp = timestamps[cam.id];
              return (
                <g key={cam.id}>
                  {/* Node circle */}
                  <rect
                    x={cx - 45}
                    y={25}
                    width={90}
                    height={60}
                    rx={10}
                    className={`camera-node ${isActive ? 'active' : ''}`}
                  />
                  {/* Camera name */}
                  <text x={cx} y={48} className="camera-node-name" textAnchor="middle">
                    {cam.name}
                  </text>
                  {/* Camera label */}
                  <text x={cx} y={65} className="camera-node-label" textAnchor="middle">
                    {cam.label}
                    {cam.sublabel ? ` (${cam.sublabel})` : ''}
                  </text>
                  {/* Timestamp below node */}
                  {isActive && camTimestamp && (
                    <text
                      x={cx}
                      y={105}
                      className="camera-node-time"
                      textAnchor="middle"
                    >
                      {formatTimeOnly(camTimestamp)}
                    </text>
                  )}
                  {/* Order indicator */}
                  {isActive && (
                    <circle
                      cx={cx + 35}
                      cy={30}
                      r={10}
                      className="order-badge"
                    />
                  )}
                  {isActive && (
                    <text
                      x={cx + 35}
                      y={34}
                      className="order-badge-text"
                      textAnchor="middle"
                    >
                      {passedCameras.indexOf(cam.id) + 1}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {!selectedVehicle && (
            <div className="route-map-placeholder">
              Click a vehicle row to see its camera route
            </div>
          )}
        </div>

        {/* Selected vehicle details */}
        {selectedVehicle && (
          <div className="vehicle-details-panel">
            <div className="detail-item">
              <span className="detail-label">Vehicle ID</span>
              <span className="detail-value mono">{selectedVehicle.tracking_id}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Vehicle</span>
              <span className="detail-value">
                {selectedVehicle.color || 'Unknown'} {selectedVehicle.car_name || 'Unknown'}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Plate</span>
              <span className="detail-value mono">
                {selectedVehicle.plate_number || 'N/A'}
                {selectedVehicle.plate_confidence != null && selectedVehicle.plate_number && (
                  <span className="plate-conf">
                    {' '}({(selectedVehicle.plate_confidence * 100).toFixed(0)}%)
                  </span>
                )}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Cameras</span>
              <span className="detail-value">
                {selectedVehicle.cameras_passed.length} of {CAMERA_INFO.length}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Duration</span>
              <span className="detail-value">
                {formatTimestamp(selectedVehicle.first_seen_time)} &mdash; {formatTimestamp(selectedVehicle.last_seen_time)}
              </span>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <Layout>
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="vehicle-log-container">
        <h1 className="page-title">Vehicle Detection Log</h1>

        {/* Camera Route Map */}
        {renderRouteMap()}

        {/* Controls */}
        <div className="log-controls">
          <div className="search-bar">
            <input
              type="text"
              placeholder="Search by plate, color, or vehicle name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showActiveOnly}
              onChange={(e) => setShowActiveOnly(e.target.checked)}
            />
            <span>Active Only</span>
          </label>
          <div className="log-count">
            {filteredData.length} vehicle{filteredData.length !== 1 ? 's' : ''}
          </div>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        {/* Vehicle Log Table */}
        <div className="card">
          <div className="card-header">Detection Records</div>
          {filteredData.length === 0 ? (
            <div className="no-data">
              {searchQuery ? 'No vehicles match your search' : 'No vehicle detections recorded'}
            </div>
          ) : (
            <div className="table-container">
              <table className="table vehicle-log-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Vehicle ID</th>
                    <th>Vehicle Name</th>
                    <th>Color</th>
                    <th>Plate Number</th>
                    <th>Confidence</th>
                    <th>Camera Route</th>
                    <th>First Seen</th>
                    <th>Last Seen</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((record, index) => (
                    <tr
                      key={record.id}
                      className={`log-row ${record.is_active ? 'active-vehicle' : 'inactive-vehicle'} ${selectedVehicle?.id === record.id ? 'selected-row' : ''}`}
                      onClick={() =>
                        setSelectedVehicle(
                          selectedVehicle?.id === record.id ? null : record
                        )
                      }
                    >
                      <td className="sno">{index + 1}</td>
                      <td className="vehicle-id">
                        {record.tracking_id.substring(0, 8)}...
                      </td>
                      <td className="car-name">{record.car_name || 'Unknown'}</td>
                      <td className="color-cell">
                        <span
                          className="color-dot"
                          style={{
                            backgroundColor: getColorHex(record.color),
                          }}
                        ></span>
                        {record.color || 'Unknown'}
                      </td>
                      <td className="plate-number">
                        {record.plate_number || <span className="no-plate">N/A</span>}
                      </td>
                      <td>
                        {record.confidence_score != null ? (
                          <span className="confidence-score">
                            {record.confidence_score.toFixed(1)}%
                          </span>
                        ) : (
                          <span className="no-plate">N/A</span>
                        )}
                      </td>
                      <td>{renderCameraRoute(record.cameras_passed)}</td>
                      <td className="timestamp">{formatTimestamp(record.first_seen_time)}</td>
                      <td className="timestamp">{formatTimestamp(record.last_seen_time)}</td>
                      <td>
                        <span
                          className={`status-badge ${record.is_active ? 'active' : 'inactive'}`}
                        >
                          {record.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="log-footer">
            <small>Auto-refreshing every 5 seconds</small>
          </div>
        </div>
      </div>
    </Layout>
  );
};

function getColorHex(colorName) {
  if (!colorName) return '#999';
  const colors = {
    white: '#f0f0f0',
    black: '#333',
    red: '#e74c3c',
    blue: '#3498db',
    green: '#27ae60',
    silver: '#bdc3c7',
    gray: '#7f8c8d',
    grey: '#7f8c8d',
    yellow: '#f1c40f',
    orange: '#e67e22',
    brown: '#8b4513',
    gold: '#ffd700',
    beige: '#f5f5dc',
    maroon: '#800000',
  };
  return colors[colorName.toLowerCase()] || '#999';
}

export default VehicleLog;
