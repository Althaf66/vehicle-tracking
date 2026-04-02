import { useState, useEffect } from 'react';
import { trackingAPI } from '../services/api';
import './VehicleDetectionLog.css';

const VehicleDetectionLog = () => {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchRecords();
    const interval = setInterval(fetchRecords, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchRecords = async () => {
    try {
      const response = await trackingAPI.getAll();
      const sorted = [...response.data].sort(
        (a, b) => new Date(b.last_seen_time) - new Date(a.last_seen_time)
      );
      setRecords(sorted);
      setError('');
    } catch (err) {
      setError('Failed to fetch detection log');
      console.error(err);
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour12: false });
  };

  return (
    <div className="detection-log-card">
      <div className="detection-log-header">
        <span>Vehicle Detection Log</span>
        <span className="detection-log-badge">{records.length} records</span>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="detection-log-container">
        {records.length === 0 ? (
          <div className="detection-log-empty">No detections recorded</div>
        ) : (
          records.map((record) => (
            <div key={record.id} className="detection-log-entry">
              <span className="log-time">[{formatTime(record.last_seen_time)}]</span>
              <span className="log-text">
                Vehicle detected: {record.color || 'Unknown'} {record.car_name || 'Unknown'}
                {' | '}Plate: {record.plate_number || 'N/A'}
                {' | '}Camera: {record.cameras_passed?.[record.cameras_passed.length - 1] || 'N/A'}
                {record.confidence_score != null && (
                  <>{' | '}Confidence: {record.confidence_score.toFixed(1)}%</>
                )}
              </span>
            </div>
          ))
        )}
      </div>

      <div className="detection-log-footer">
        <small>Auto-refreshing every 5 seconds</small>
      </div>
    </div>
  );
};

export default VehicleDetectionLog;
