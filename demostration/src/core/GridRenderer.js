export class GridRenderer {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
        this.cells = [];
    }
    
    createModelInputGrid() {
        const container = document.getElementById('modelInputGridContainer');
        if (!container) {
            console.error("Container 'modelInputGridContainer' not found.");
            return;
        }
        
        container.innerHTML = '';
        
        const grid = document.createElement('div');
        grid.className = 'model-input-grid';
        
        const gridSize = this.drawingGrid.settings.gridSize;
        grid.style.gridTemplateColumns = `repeat(${gridSize}, 1fr)`;
        
        this.cells = [];

        for (let i = 0; i < gridSize * gridSize; i++) {
            const cell = document.createElement('div');
            
            cell.style.backgroundColor = 'rgb(0, 0, 0)'; 
            cell.dataset.index = i;
            
            this.cells.push(cell);
            grid.appendChild(cell);
        }
        
        container.appendChild(grid);
        this.updateModelInputGrid(this.drawingGrid.gridData);
    }
    
    updateModelInputGrid(gridData) {
        const invert = this.drawingGrid.settings.invertColors;

        gridData.forEach((value, index) => {
            const cell = this.cells[index];
            if (!cell) return;
            
            let brightness;
            
            if (invert) {
                brightness = value;
            } else {
                brightness = 255 - value;
            }

            cell.style.backgroundColor = `rgb(${brightness}, ${brightness}, ${brightness})`;
        });
    }
}
