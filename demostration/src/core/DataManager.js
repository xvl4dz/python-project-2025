export class DataManager {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
    }
    
    clearGrid() {
        this.drawingGrid.gridData.fill(0);
        
        for (let i = 0; i < this.drawingGrid.gridData.length; i++) {
            this.drawingGrid.updateCellAppearance(i, this.drawingGrid.gridData[i]);
        }
        
        this.drawingGrid.serverClient.clearServer();
        this.drawingGrid.updateInfo();
    }
    
    displayData(dataArray) {
        if (!dataArray || !Array.isArray(dataArray)) {
            console.error('Invalid data array provided');
            return false;
        }
        
        let flatArray;
        if (Array.isArray(dataArray[0])) {
            flatArray = dataArray.flat();
        } else {
            flatArray = dataArray;
        }
        
        if (flatArray.length !== this.drawingGrid.settings.gridSize * this.drawingGrid.settings.gridSize) {
            console.error(`Array size ${flatArray.length} doesn't match grid size ${this.drawingGrid.settings.gridSize * this.drawingGrid.settings.gridSize}`);
            return false;
        }
        
        for (let i = 0; i < flatArray.length; i++) {
            const value = flatArray[i];
            if (typeof value !== 'number' || value < 0 || value > 255) {
                console.error(`Invalid value at index ${i}: ${value}. Must be number between 0-255`);
                return false;
            }
        }
        
        this.drawingGrid.gridData = [...flatArray];
        
        for (let i = 0; i < this.drawingGrid.gridData.length; i++) {
            this.drawingGrid.updateCellAppearance(i, this.drawingGrid.gridData[i]);
        }
        
        this.drawingGrid.serverClient.updateServer();
        
        console.log('Data loaded successfully');
        return true;
    }
    
    promptLoadData() {
        const dataStr = prompt('Enter array data as JSON (1D or 2D array with values 0-255):');
        if (dataStr) {
            try {
                const dataArray = JSON.parse(dataStr);
                if (this.displayData(dataArray)) {
                    alert('Data loaded successfully!');
                } else {
                    alert('Failed to load data. Check console for details.');
                }
            } catch (e) {
                alert('Invalid JSON format: ' + e.message);
            }
        }
    }
    
    loadExampleData() {
        const exampleData = Array(this.drawingGrid.settings.gridSize * this.drawingGrid.settings.gridSize).fill(0);
        const center = Math.floor(this.drawingGrid.settings.gridSize / 2);
        for (let i = 0; i < this.drawingGrid.settings.gridSize; i++) {
            exampleData[center * this.drawingGrid.settings.gridSize + i] = 255;
            exampleData[i * this.drawingGrid.settings.gridSize + center] = 255;
        }
        
        this.displayData(exampleData);
        console.log('Example data loaded');
    }
    
    showGridData() {
        console.log('Grid Data (N×N array):');
        
        const grid2D = this.drawingGrid.getGridData2D();
        console.table(grid2D);
        
        const info = document.getElementById('info');
        info.innerHTML = `
            <strong>Grid Data (first few values):</strong><br>
            ${this.drawingGrid.gridData.slice(0, 10).join(', ')}...<br>
            <em>Check browser console for full N×N array</em>
        `;
    }
    
    exportGridData() {
        const dataStr = JSON.stringify(this.drawingGrid.getGridData2D());
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `mnist_like_data_${this.drawingGrid.settings.gridSize}x${this.drawingGrid.settings.gridSize}.json`;
        link.click();
        
        const info = document.getElementById('info');
        info.innerHTML = `Data exported as ${link.download}`;
    }
}