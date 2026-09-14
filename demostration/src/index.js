import { DrawingGrid } from './core/DrawingGrid.js';

document.addEventListener('DOMContentLoaded', () => {
    window.drawingGrid = new DrawingGrid();
    
    window.displayGridData = (dataArray) => {
        return window.drawingGrid.displayData(dataArray);
    };
});