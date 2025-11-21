export class ServerClient {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
    }
    
    async testConnection() {
        try {
            const response = await fetch(`${this.drawingGrid.settings.serverUrl}/api/info`);
            if (response.ok) {
                const data = await response.json();
                console.log('Server connected:', data);
                this.drawingGrid.updateInfo('Server: Connected');
            } else {
                console.warn('Server not responding');
                this.drawingGrid.updateInfo('Server: Not connected');
            }
        } catch (error) {
            console.warn('Cannot connect to server:', error);
            this.drawingGrid.updateInfo('Server: Not connected');
        }
    }
    
    async updateServer() {
        try {
            const response = await fetch(`${this.drawingGrid.settings.serverUrl}/api/update`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gridData: this.drawingGrid.gridData,
                    gridSize: this.drawingGrid.settings.gridSize
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Server updated:', data.message);
            } else {
                console.error('Failed to update server');
            }
        } catch (error) {
            console.error('Error updating server:', error);
        }
    }
    
    async loadFromServer() {
        if (!this.drawingGrid.settings.serverEnabled) {
            alert('Server is not enabled');
            return;
        }
        
        try {
            const response = await fetch(`${this.drawingGrid.settings.serverUrl}/api/drawing`);
            if (response.ok) {
                const data = await response.json();
                this.drawingGrid.displayData(data.gridData);
                console.log('Loaded from server:', data.message);
                this.drawingGrid.updateInfo('Loaded from server');
            } else {
                alert('No drawing available on server');
            }
        } catch (error) {
            console.error('Error loading from server:', error);
            alert('Error loading from server: ' + error.message);
        }
    }
    
    async saveToServer() {
        if (!this.drawingGrid.settings.serverEnabled) {
            alert('Server is not enabled');
            return;
        }
        
        try {
            const response = await fetch(`${this.drawingGrid.settings.serverUrl}/api/save`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const data = await response.json();
                alert(`Drawing saved as: ${data.filename}`);
                console.log('Saved to server:', data.message);
            } else {
                const error = await response.json();
                alert('Error saving: ' + error.error);
            }
        } catch (error) {
            console.error('Error saving to server:', error);
            alert('Error saving to server: ' + error.message);
        }
    }
    
    async clearServer() {
        if (!this.drawingGrid.settings.serverEnabled) {
            alert('Server is not enabled');
            return;
        }
        
        try {
            const response = await fetch(`${this.drawingGrid.settings.serverUrl}/api/clear`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('Server drawing cleared');
                this.drawingGrid.updateInfo('Server drawing cleared');
            }
        } catch (error) {
            console.error('Error clearing server:', error);
        }
    }
}