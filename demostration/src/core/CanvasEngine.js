export class CanvasEngine {
    constructor(drawingGrid) {
        this.drawingGrid = drawingGrid;
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
    }

    initializeCanvas() {
        this.canvas = document.getElementById('drawingCanvas');
        this.canvas.width = this.drawingGrid.settings.canvasSize;
        this.canvas.height = this.drawingGrid.settings.canvasSize;

        this.ctx = this.canvas.getContext('2d');
        
        this.applyStyle();
        this.addCanvasEvents();
        this.clearCanvas();
    }
    
    applyStyle() {
        const settings = this.drawingGrid.settings;
        
        const bgColor = settings.invertColors ? '#000000' : '#ffffff';
        this.canvas.style.backgroundColor = bgColor;
        
        this.ctx.strokeStyle = settings.invertColors ? '#ffffff' : '#000000';
        
        this.ctx.lineWidth = settings.stylusSize;
        this.ctx.globalAlpha = settings.stylusSoftness;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }
    
    clearCanvas() {
        // Must reset alpha to 1.0 before clearing
        this.ctx.globalAlpha = 1.0; 
        
        this.ctx.fillStyle = this.drawingGrid.settings.invertColors ? '#000000' : '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.globalAlpha = this.drawingGrid.settings.stylusSoftness;
    }
    
    addCanvasEvents() {
        const events = {
            start: (e) => {
                e.preventDefault();
                this.isDrawing = true;
                [this.lastX, this.lastY] = this.getCoords(e);
                this.applyStyle(); 
                this.draw(e);
            },
            move: (e) => {
                if (!this.isDrawing) return;
                e.preventDefault();
                this.draw(e);
            },
            end: () => {
                if (this.isDrawing) {
                    this.isDrawing = false;
                    this.drawingGrid.processDrawingUpdate(); 
                }
            }
        };

        this.canvas.addEventListener('mousedown', events.start);
        this.canvas.addEventListener('mousemove', events.move);
        document.addEventListener('mouseup', events.end); 
        
        this.canvas.addEventListener('touchstart', events.start);
        this.canvas.addEventListener('touchmove', events.move);
        document.addEventListener('touchend', events.end);
    }
    
    getCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        let clientX, clientY;
        
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return [clientX - rect.left, clientY - rect.top];
    }

    draw(e) {
        const [x, y] = this.getCoords(e);
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        
        [this.lastX, this.lastY] = [x, y];
    }
    
    downsampleCanvas() {
        const { gridSize, canvasSize, invertColors } = this.drawingGrid.settings;
        const squareSize = canvasSize / gridSize; 
        const newGridData = Array(gridSize * gridSize).fill(0);
        
        const imageData = this.ctx.getImageData(0, 0, canvasSize, canvasSize).data;

        for (let i = 0; i < gridSize; i++) { // Row (Y)
            for (let j = 0; j < gridSize; j++) { // Column (X)
                let totalInkIntensity = 0;
                let pixelCount = 0;

                // Iterate over the high-res pixels within the current square
                for (let y = 0; y < squareSize; y++) {
                    for (let x = 0; x < squareSize; x++) {
                        
                        const pixelX = j * squareSize + x;
                        const pixelY = i * squareSize + y;
                        const index = (pixelY * canvasSize + pixelX) * 4; // *4 for RGBA
                        
                        const red = imageData[index]; 
                        
                        let inkIntensity;
                        
                        // Calculate Ink Intensity (0=Empty, 255=Full Ink)
                        if (invertColors) {
                             // White ink on Black BG. Ink is represented by high RGB value.
                             inkIntensity = red; 
                        } else {
                            // Black ink on White BG. Ink is represented by low RGB value.
                            inkIntensity = 255 - red; 
                        }

                        totalInkIntensity += inkIntensity;
                        pixelCount++;
                    }
                }

                // Average the ink intensity over the square
                const averageInk = Math.round(totalInkIntensity / pixelCount);
                newGridData[i * gridSize + j] = averageInk;
            }
        }
        
        return newGridData;
    }
}
