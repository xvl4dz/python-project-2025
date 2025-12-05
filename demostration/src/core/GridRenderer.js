export class GridRenderer {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
    }
    
    createGrid() {
        const container = document.getElementById('gridContainer');
        container.innerHTML = '';
        
        this.drawingGrid.gridData = Array(this.drawingGrid.settings.gridSize * this.drawingGrid.settings.gridSize).fill(0);
        
        const grid = document.createElement('div');
        grid.className = 'grid';
        grid.style.gridTemplateColumns = `repeat(${this.drawingGrid.settings.gridSize}, 1fr)`;
        grid.style.width = '500px';
        grid.style.height = '500px';
        
        this.applyGridStyle(grid);
        
        for (let i = 0; i < this.drawingGrid.settings.gridSize * this.drawingGrid.settings.gridSize; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.index = i;
            cell.dataset.row = Math.floor(i / this.drawingGrid.settings.gridSize);
            cell.dataset.col = i % this.drawingGrid.settings.gridSize;
            
            this.updateCellAppearance(i, this.drawingGrid.gridData[i]);
            
            if (this.drawingGrid.settings.mode === 'draw') {
                this.addDrawingEvents(cell, i);
            } else {
                cell.style.cursor = 'default';
            }
            
            grid.appendChild(cell);
        }
        
        container.appendChild(grid);
        
        if (this.drawingGrid.settings.mode === 'draw') {
            document.addEventListener('mouseup', () => {
                this.drawingGrid.isDrawing = false;
            });
        }
        
        grid.addEventListener('dragstart', (e) => {
            e.preventDefault();
        });
        
        this.drawingGrid.updateInfo();
    }
    
    addDrawingEvents(cell, index) {
        cell.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.drawingGrid.isDrawing = true;
            this.drawingGrid.applyStylus(index);
        });
        
        cell.addEventListener('mouseenter', () => {
            if (this.drawingGrid.isDrawing) {
                this.drawingGrid.applyStylus(index);
            }
        });
        
        cell.addEventListener('click', () => {
            this.drawingGrid.applyStylus(index);
        });
    }
    
    applyGridStyle(gridElement = null) {
        const grid = gridElement || document.querySelector('.grid');
        if (!grid) return;
        
        if (!this.drawingGrid.settings.showGrid) {
            grid.style.gap = '0px';
            grid.style.backgroundColor = this.drawingGrid.settings.invertColors ? 'black' : '#ccc';
        } else {
            grid.style.gap = '1px';
            grid.style.backgroundColor = '#ccc';
        }
    }
    
    updateCellAppearance(index, value) {
        const cell = document.querySelector(`.cell[data-index="${index}"]`);
        if (cell) {
            if (this.drawingGrid.settings.invertColors) {
                cell.style.backgroundColor = `rgb(${value}, ${value}, ${value})`;
                cell.title = `Value: ${value}/255`;
            } else {
                const brightness = 255 - value;
                cell.style.backgroundColor = `rgb(${brightness}, ${brightness}, ${brightness})`;
                cell.title = `Value: ${value}/255`;
            }
        }
    }
}