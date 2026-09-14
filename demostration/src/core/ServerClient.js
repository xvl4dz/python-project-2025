export class ServerClient {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
        this.dataManager = drawingGrid.dataManager;
        this.url = drawingGrid.settings.serverUrl;
    }

    testConnection() {
        this.drawingGrid.updateInfo('Connecting...');
        
        fetch(this.url + '/api/info') 
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const modelStatus = data.model_loaded ? 'Model Loaded' : 'No Model';
                this.drawingGrid.updateInfo(`Connected: ${modelStatus}`);
                
                if (this.drawingGrid.initializePredictionUI) {
                    this.drawingGrid.initializePredictionUI(data.class_labels); 
                }
                
                this.loadFromServer(true);
            })
            .catch(error => {
                console.error("Connection failed:", error);
                this.drawingGrid.updateInfo('Connection Failed');
            });
    }

    updateServer() {
        if (!this.drawingGrid.settings.serverEnabled) return;
        
        if (!this.drawingGrid.dataManager) {
            console.error("DataManager not available, skipping server update.");
            return;
        }

        const dataPayload = {
            gridData: this.drawingGrid.dataManager.getGridData2D(),
            gridSize: this.drawingGrid.settings.gridSize
        };

        fetch(this.url + '/api/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataPayload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (this.drawingGrid.updatePredictionDisplay) {
                this.drawingGrid.updatePredictionDisplay(data);
            }
        })
        .catch(error => {
            console.error("Prediction update failed:", error);
        });
    }

    loadFromServer(isInitialLoad = false) {
        if (!this.drawingGrid.dataManager) return;
        this.drawingGrid.updateInfo(isInitialLoad ? 'Checking for saved data...' : 'Pulling data...');

        fetch(this.url + '/api/drawing')
            .then(response => {
                if (response.status === 404) {
                    if (!isInitialLoad) {
                        this.drawingGrid.updateInfo('No saved drawing found.');
                    }
                    return null;
                }
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data && data.gridData) {
                    this.drawingGrid.dataManager.displayData(data.gridData);
                    this.drawingGrid.updateInfo(isInitialLoad ? 'Initial data loaded.' : 'Data loaded successfully');
                } else if (data) {
                    console.warn('Received data but missing gridData key:', data);
                }
            })
            .catch(error => {
                console.error("Load failed:", error);
                this.drawingGrid.updateInfo('Load Failed');
            });
    }

    saveToServer() {
        if (!this.drawingGrid.dataManager) return;
        this.drawingGrid.updateInfo('Pushing data...');
        
        const dataPayload = {
            gridData: this.drawingGrid.dataManager.getGridData2D(),
            gridSize: this.drawingGrid.settings.gridSize
        };

        fetch(this.url + '/api/save', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataPayload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            this.drawingGrid.updateInfo('Data saved successfully');
        })
        .catch(error => {
            console.error("Save failed:", error);
            this.drawingGrid.updateInfo('Save Failed');
        });
    }
}
