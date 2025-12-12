import { GridRenderer } from './GridRenderer.js';
import { ServerClient } from './ServerClient.js';
import { DataManager } from './DataManager.js';
import { CanvasEngine } from './CanvasEngine.js'; 

export class DrawingGrid {
    constructor() {
        this.settings = {
            gridSize: 28,
            canvasSize: 280, 
            invertColors: true, 
            serverEnabled: true,
            serverUrl: 'http://localhost:5000',
            
            stylusSize: 15,
            stylusSoftness: 1.0
        };
        
        this.gridData = Array(this.settings.gridSize * this.settings.gridSize).fill(0); 
        this.probabilityBars = {};
        this.isDrawing = false;
        this.gui = null;
        
        this.gridRenderer = new GridRenderer(this); 
        this.canvasEngine = new CanvasEngine(this); 
        this.dataManager = new DataManager(this); // Initialize DataManager before ServerClient
        this.serverClient = new ServerClient(this);
        
        this.initializeGUI(); 
        this.initializeEventListeners(); 

        this.canvasEngine.initializeCanvas();
        this.gridRenderer.createModelInputGrid();
        
        if (this.settings.serverEnabled) {
            this.serverClient.testConnection();
        }
    }
    
    initializeGUI() {
        if (window.location.hash !== '#dev') return;
        
        this.gui = new dat.GUI();
        
        const mainFolder = this.gui.addFolder('Main Settings');
        mainFolder.add(this.settings, 'serverEnabled').name('Server Enabled');
        
        const canvasFolder = this.gui.addFolder('Canvas/Grid Settings');
        canvasFolder.add(this.settings, 'invertColors').name('Invert Colors')
            .onChange(() => this.toggleColorsWithClear());
        
        const stylusFolder = this.gui.addFolder('Stylus Settings');
        stylusFolder.add(this.settings, 'stylusSize', 1, 30).name('Size (px)')
            .onChange(() => this.canvasEngine.applyStyle());
        stylusFolder.add(this.settings, 'stylusSoftness', 0.1, 1.0).name('Softness (Alpha)')
            .onChange(() => this.canvasEngine.applyStyle());

        mainFolder.open();
        canvasFolder.open();
        stylusFolder.open();
    }
    
    initializeEventListeners() {
        document.getElementById('clearGrid').addEventListener('click', () => {
            this.clearGrid();
        });
        document.getElementById('toggleColors').addEventListener('click', () => this.toggleColorsWithClear());
        
        document.getElementById('serverLoad').addEventListener('click', () => this.serverClient.loadFromServer());
        document.getElementById('serverSave').addEventListener('click', () => this.serverClient.saveToServer());
    }

    processDrawingUpdate() {
        this.gridData = this.canvasEngine.downsampleCanvas();
        this.gridRenderer.updateModelInputGrid(this.gridData); 
        this.triggerServerUpdate();
    }
    
    clearGrid() { 
        this.canvasEngine.clearCanvas();
        this.gridData.fill(0);
        this.gridRenderer.updateModelInputGrid(this.gridData);
        
        this.updatePredictionDisplay({ 
            prediction: '?', 
            confidence: 0, 
            probabilities: Array(10).fill(0),
            labels: Object.keys(this.probabilityBars) 
        });
    }
    
    toggleColorsWithClear() {
        this.settings.invertColors = !this.settings.invertColors;
        this.canvasEngine.applyStyle();
        this.clearGrid();
        this.triggerServerUpdate(); 
    }

    // Called by ServerClient when connected to generate the progress bars
    initializePredictionUI(labels) {
        const container = document.getElementById('probsContainer');
        if (!container) return;

        container.innerHTML = '';
        this.probabilityBars = {}; 

        labels.forEach(label => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            
            row.innerHTML = `
                <span class="prob-label">${label}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width: 0%"></div>
                </div>
                <span class="prob-value">0%</span>
            `;
            
            container.appendChild(row);
            
            this.probabilityBars[label] = {
                fill: row.querySelector('.prob-fill'),
                value: row.querySelector('.prob-value')
            };
        });
    }

    // Called by ServerClient on update
    updatePredictionDisplay(data) {
        if (!data) return;

        const predBig = document.getElementById('predictionBig');
        const confBig = document.getElementById('confidenceBig');
        
        if (predBig) predBig.innerText = data.prediction || '?';
        if (confBig) confBig.innerText = data.confidence 
            ? `${(data.confidence * 100).toFixed(1)}% Confidence` 
            : 'Waiting...';

        if (data.probabilities && data.labels) {
            data.labels.forEach((label, index) => {
                const prob = data.probabilities[index] || 0;
                const percent = (prob * 100).toFixed(1) + '%';
                
                if (this.probabilityBars[label]) {
                    this.probabilityBars[label].fill.style.width = percent;
                    this.probabilityBars[label].value.innerText = percent;
                    
                    if (prob > 0.5) {
                        this.probabilityBars[label].fill.classList.add('high');
                    } else {
                        this.probabilityBars[label].fill.classList.remove('high');
                    }
                }
            });
        }
    }

    displayData(data) {
        this.dataManager.displayData(data);
    }

    triggerServerUpdate() { 
        this.serverClient.updateServer(); 
    }
    
    updateInfo(msg) { 
        const infoEl = document.getElementById('info');
        if (infoEl) infoEl.innerHTML = msg || 'Ready'; 
    }
}
