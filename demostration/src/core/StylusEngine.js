export class StylusEngine {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
        this.lastUpdateTime = 0;
        this.updateDebounce = 100;
    }
    
    applyStylus(centerIndex) {
        if (this.drawingGrid.settings.mode !== 'draw') return;
        
        const affectedCells = this.getAffectedCells(centerIndex);
        
        affectedCells.forEach(cell => {
            const currentValue = this.drawingGrid.gridData[cell.index];
            const valueChange = Math.round(255 * cell.intensity);
            let newValue;
            
            if (this.drawingGrid.settings.invertColors) {
                newValue = Math.min(255, currentValue + valueChange);
            } else {
                newValue = Math.min(255, currentValue + valueChange);
            }
            
            this.drawingGrid.gridData[cell.index] = newValue;
            this.drawingGrid.updateCellAppearance(cell.index, newValue);
        });
        
        this.updateServer();
    }
    
    getAffectedCells(centerIndex) {
        const centerRow = Math.floor(centerIndex / this.drawingGrid.settings.gridSize);
        const centerCol = centerIndex % this.drawingGrid.settings.gridSize;
        const affectedCells = [];
        const radius = this.drawingGrid.settings.stylusSize;
        
        const minRow = Math.max(0, Math.floor(centerRow - radius));
        const maxRow = Math.min(this.drawingGrid.settings.gridSize - 1, Math.ceil(centerRow + radius));
        const minCol = Math.max(0, Math.floor(centerCol - radius));
        const maxCol = Math.min(this.drawingGrid.settings.gridSize - 1, Math.ceil(centerCol + radius));
        
        for (let row = minRow; row <= maxRow; row++) {
            for (let col = minCol; col <= maxCol; col++) {
                const distance = this.calculateDistance(centerRow, centerCol, row, col);
                
                if (this.drawingGrid.settings.stylusShape === 'circle' && distance > radius) {
                    continue;
                }
                
                let intensity = 1;
                if (this.drawingGrid.settings.stylusSoftness > 0 && distance > 0) {
                    const normalizedDistance = distance / radius;
                    intensity = Math.max(0, 1 - (normalizedDistance / this.drawingGrid.settings.stylusSoftness));
                }
                
                const index = row * this.drawingGrid.settings.gridSize + col;
                affectedCells.push({
                    index: index,
                    intensity: intensity
                });
            }
        }
        
        return affectedCells;
    }
    
    calculateDistance(row1, col1, row2, col2) {
        return Math.sqrt(Math.pow(row2 - row1, 2) + Math.pow(col2 - col1, 2));
    }
    
    async updateServer() {
        if (!this.drawingGrid.settings.serverEnabled) return;
        
        const now = Date.now();
        if (now - this.lastUpdateTime < this.updateDebounce) {
            return;
        }
        this.lastUpdateTime = now;
        
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
}