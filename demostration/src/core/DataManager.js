export class DataManager {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
    }

    displayData(data) {
        if (!data || data.length === 0) {
            console.warn("Received empty data array.");
            return;
        }

        const grid = this.drawingGrid;
        const ctx = grid.canvasEngine.ctx;
        const { gridSize, canvasSize, invertColors } = grid.settings;
        const squareSize = canvasSize / gridSize;

        // 1. Update the internal grid data (28x28)
        grid.gridData = data;
        
        // 2. Update the visual 28x28 model input grid
        grid.gridRenderer.updateModelInputGrid(data);

        // 3. Update the High-Resolution Canvas
        
        grid.canvasEngine.clearCanvas();
        grid.canvasEngine.applyStyle(); 

        const imageData = ctx.createImageData(canvasSize, canvasSize);
        const dataArray = imageData.data;
        
        for (let i = 0; i < gridSize; i++) { // 28 rows
            for (let j = 0; j < gridSize; j++) { // 28 columns
                const gridIndex = i * gridSize + j;
                
                // Ink Intensity (0=Empty, 255=Full Ink)
                const inkIntensity = data[gridIndex]; 

                let brightness; // This is the actual value written to the canvas R/G/B channels
                
                if (invertColors) {
                    // White Ink on Black BG
                    brightness = inkIntensity;
                } else {
                    // Black Ink on White BG
                    brightness = 255 - inkIntensity;
                }
                
                // Loop over the 10x10 high-res pixels corresponding to this 28x28 cell
                for (let y = 0; y < squareSize; y++) {
                    for (let x = 0; x < squareSize; x++) {
                        
                        const pixelX = j * squareSize + x;
                        const pixelY = i * squareSize + y;
                        const index = (pixelY * canvasSize + pixelX) * 4;

                        dataArray[index] = brightness;     // R
                        dataArray[index + 1] = brightness; // G
                        dataArray[index + 2] = brightness; // B
                        dataArray[index + 3] = 255;        // A
                    }
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);

        // 4. Trigger server update (Optional, but good practice if loading triggers re-prediction)
        grid.triggerServerUpdate();
    }

    getGridData2D() {
        return this.drawingGrid.gridData; 
    }
}