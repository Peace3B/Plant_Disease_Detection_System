import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [uptime, setUptime] = useState(null);
  const [classDistribution, setClassDistribution] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [retrainStatus, setRetrainStatus] = useState(null);

  useEffect(() => {
    // Fetch initial data
    fetchMetrics();
    fetchUptime();
    fetchClassDistribution();
    fetchTrainingHistory();
    
    // Set up polling for uptime
    const interval = setInterval(fetchUptime, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchUptime = async () => {
    try {
      const response = await axios.get(`${API_URL}/uptime`);
      setUptime(response.data);
    } catch (error) {
      console.error('Error fetching uptime:', error);
    }
  };

  const fetchClassDistribution = async () => {
    try {
      const response = await axios.get(`${API_URL}/visualizations/class-distribution`);
      setClassDistribution(response.data);
    } catch (error) {
      console.error('Error fetching class distribution:', error);
    }
  };

  const fetchTrainingHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/training-history`);
      setTrainingHistory(response.data);
    } catch (error) {
      console.error('Error fetching training history:', error);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPrediction(response.data);
    } catch (error) {
      alert('Prediction error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadFiles = (event) => {
    setUploadFiles(Array.from(event.target.files));
  };

  const handleBulkUpload = async () => {
    if (uploadFiles.length === 0) return;

    setLoading(true);
    const formData = new FormData();
    uploadFiles.forEach(file => formData.append('files', file));

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert(`Successfully uploaded ${response.data.uploaded} images`);
      setUploadFiles([]);
    } catch (error) {
      alert('Upload error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (!window.confirm('Start model retraining? This may take several minutes.')) {
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/retrain`, {
        epochs: 20
      });
      setRetrainStatus(response.data);
      
      // Poll for status
      const statusInterval = setInterval(async () => {
        try {
          const statusResponse = await axios.get(`${API_URL}/retrain/status`);
          setRetrainStatus(statusResponse.data);
          
          if (!statusResponse.data.is_retraining) {
            clearInterval(statusInterval);
            alert('Retraining completed!');
            fetchMetrics();
          }
        } catch (error) {
          clearInterval(statusInterval);
        }
      }, 3000);
      
    } catch (error) {
      alert('Retraining error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDiseaseName = (name) => {
    if (!name) return '';
    const parts = name.split('___');
    if (parts.length === 2) {
      return `${parts[0].replace(/_/g, ' ')} - ${parts[1].replace(/_/g, ' ')}`;
    }
    return name.replace(/_/g, ' ');
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🌿 Plant Disease Classification System</h1>
        <p>AI-Powered Plant Health Diagnostics</p>
      </header>

      <nav className="nav-tabs">
        <button 
          className={activeTab === 'predict' ? 'active' : ''}
          onClick={() => setActiveTab('predict')}
        >
          Predict
        </button>
        <button 
          className={activeTab === 'visualizations' ? 'active' : ''}
          onClick={() => setActiveTab('visualizations')}
        >
          Visualizations
        </button>
        <button 
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Upload & Retrain
        </button>
        <button 
          className={activeTab === 'monitoring' ? 'active' : ''}
          onClick={() => setActiveTab('monitoring')}
        >
          Monitoring
        </button>
      </nav>

      <div className="content">
        {/* PREDICT TAB */}
        {activeTab === 'predict' && (
          <div className="tab-content">
            <h2>Plant Disease Prediction</h2>
            
            <div className="upload-section">
              <label className="file-input-label">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileSelect}
                  className="file-input"
                />
                {selectedFile ? selectedFile.name : 'Choose an image...'}
              </label>
              
              {preview && (
                <div className="preview">
                  <img src={preview} alt="Preview" />
                </div>
              )}
              
              <button 
                onClick={handlePredict} 
                disabled={!selectedFile || loading}
                className="btn-primary"
              >
                {loading ? 'Analyzing...' : 'Predict Disease'}
              </button>
            </div>

            {prediction && (
              <div className="prediction-result">
                <h3>Prediction Results</h3>
                <div className="top-prediction">
                  <h4>{formatDiseaseName(prediction.top_prediction)}</h4>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                  <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
                  <p className="processing-time">
                    Processing time: {(prediction.processing_time * 1000).toFixed(0)}ms
                  </p>
                </div>

                <h4>Alternative Predictions:</h4>
                <div className="predictions-list">
                  {prediction.predictions.slice(1).map((pred, idx) => (
                    <div key={idx} className="prediction-item">
                      <span>{formatDiseaseName(pred.class)}</span>
                      <span className="confidence-badge">
                        {(pred.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* VISUALIZATIONS TAB */}
        {activeTab === 'visualizations' && (
          <div className="tab-content">
            <h2>Data Visualizations</h2>
            
            {/* Training History */}
            {trainingHistory && (
              <div className="viz-section">
                <h3>Training History</h3>
                <div className="chart-container">
                  <h4>Model Accuracy Over Epochs</h4>
                  <div className="simple-chart">
                    {trainingHistory.history.accuracy && (
                      <p>
                        Final Training Accuracy: {(trainingHistory.history.accuracy[trainingHistory.history.accuracy.length - 1] * 100).toFixed(2)}%
                      </p>
                    )}
                    {trainingHistory.history.val_accuracy && (
                      <p>
                        Final Validation Accuracy: {(trainingHistory.history.val_accuracy[trainingHistory.history.val_accuracy.length - 1] * 100).toFixed(2)}%
                      </p>
                    )}
                    <p>Total Epochs: {trainingHistory.epochs}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Class Distribution */}
            {classDistribution && (
              <div className="viz-section">
                <h3>Dataset Class Distribution</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <h4>Total Images</h4>
                    <p className="stat-value">{classDistribution.total_images}</p>
                  </div>
                  <div className="stat-card">
                    <h4>Number of Classes</h4>
                    <p className="stat-value">{classDistribution.num_classes}</p>
                  </div>
                  <div className="stat-card">
                    <h4>Avg per Class</h4>
                    <p className="stat-value">
                      {Math.round(classDistribution.total_images / classDistribution.num_classes)}
                    </p>
                  </div>
                </div>
                
                <div className="class-list">
                  <h4>Images per Class (Top 10)</h4>
                  {Object.entries(classDistribution.distribution)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 10)
                    .map(([className, count]) => (
                      <div key={className} className="class-bar">
                        <span className="class-name">{formatDiseaseName(className)}</span>
                        <div className="bar">
                          <div 
                            className="bar-fill" 
                            style={{ width: `${(count / Math.max(...Object.values(classDistribution.distribution))) * 100}%` }}
                          ></div>
                        </div>
                        <span className="count">{count}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* UPLOAD & RETRAIN TAB */}
        {activeTab === 'upload' && (
          <div className="tab-content">
            <h2>Upload Data & Retrain Model</h2>
            
            <div className="upload-section">
              <h3>Bulk Image Upload</h3>
              <p className="help-text">
                Upload multiple plant images for retraining. Images should be organized by disease type.
              </p>
              
              <label className="file-input-label">
                <input 
                  type="file" 
                  accept="image/*" 
                  multiple
                  onChange={handleUploadFiles}
                  className="file-input"
                />
                {uploadFiles.length > 0 ? `${uploadFiles.length} files selected` : 'Choose images...'}
              </label>
              
              {uploadFiles.length > 0 && (
                <div className="file-list">
                  <h4>Selected Files:</h4>
                  {uploadFiles.slice(0, 10).map((file, idx) => (
                    <div key={idx} className="file-item">{file.name}</div>
                  ))}
                  {uploadFiles.length > 10 && (
                    <p>... and {uploadFiles.length - 10} more</p>
                  )}
                </div>
              )}
              
              <button 
                onClick={handleBulkUpload}
                disabled={uploadFiles.length === 0 || loading}
                className="btn-primary"
              >
                {loading ? 'Uploading...' : 'Upload Images'}
              </button>
            </div>

            <div className="retrain-section">
              <h3>Trigger Model Retraining</h3>
              <p className="help-text">
                Retrain the model with newly uploaded data. This process may take 10-30 minutes.
              </p>
              
              {retrainStatus && retrainStatus.is_retraining && (
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${retrainStatus.progress}%` }}
                  ></div>
                  <span>{retrainStatus.progress}%</span>
                </div>
              )}
              
              <button 
                onClick={handleRetrain}
                disabled={loading || (retrainStatus && retrainStatus.is_retraining)}
                className="btn-warning"
              >
                {retrainStatus && retrainStatus.is_retraining ? 'Retraining...' : 'Start Retraining'}
              </button>
            </div>
          </div>
        )}

        {/* MONITORING TAB */}
        {activeTab === 'monitoring' && (
          <div className="tab-content">
            <h2>System Monitoring</h2>
            
            {uptime && (
              <div className="monitoring-section">
                <h3>Model Uptime</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <h4>Uptime</h4>
                    <p className="stat-value">{uptime.model_uptime}</p>
                  </div>
                  <div className="stat-card">
                    <h4>Predictions Made</h4>
                    <p className="stat-value">{uptime.predictions_made}</p>
                  </div>
                  <div className="stat-card">
                    <h4>Avg Latency</h4>
                    <p className="stat-value">{uptime.average_latency_ms}ms</p>
                  </div>
                  <div className="stat-card">
                    <h4>Status</h4>
                    <p className="stat-value status-operational">{uptime.status}</p>
                  </div>
                </div>
              </div>
            )}

            {metrics && (
              <div className="monitoring-section">
                <h3>Model Metrics</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <h4>Number of Classes</h4>
                    <p className="stat-value">{metrics.num_classes}</p>
                  </div>
                  {metrics.test_accuracy && (
                    <div className="stat-card">
                      <h4>Test Accuracy</h4>
                      <p className="stat-value">
                        {(metrics.test_accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="footer">
        <p>Plant Disease Classification System © 2024</p>
      </footer>
    </div>
  );
}

export default App;