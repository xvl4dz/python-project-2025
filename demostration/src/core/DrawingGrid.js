import { GridRenderer } from './GridRenderer.js';
import { StylusEngine } from './StylusEngine.js';
import { ServerClient } from './ServerClient.js';
import { DataManager } from './DataManager.js';

export class DrawingGrid {
    constructor() {
        this.settings = {
            gridSize: 28,
            stylusSize: 2.9,
            stylusSoftness: 0.55,
            darknessIncrement: 255,
            stylusShape: 'circle',
            invertColors: true,
            showGrid: false,
            mode: 'draw',
            serverEnabled: true,
            serverUrl: 'http://localhost:5000'
        };
        
        this.gridData = [];
        this.isDrawing = false;
        this.gui = null;
        
        this.gridRenderer = new GridRenderer(this);
        this.stylusEngine = new StylusEngine(this);
        this.serverClient = new ServerClient(this);
        this.dataManager = new DataManager(this);
        
        this.initializeGUI();
        this.initializeEventListeners();
        this.createGrid();
        this.clearGrid();
        
        if (this.settings.serverEnabled) {
            this.serverClient.testConnection();
        }
    }
    
    initializeGUI() {
        this.gui = new dat.GUI();
        
        const gridFolder = this.gui.addFolder('Grid Settings');
        gridFolder.add(this.settings, 'gridSize', 2, 100, 1).name('Grid Size').onChange(() => this.createGrid());
        gridFolder.add(this.settings, 'showGrid').name('Show Grid Lines').onChange(() => this.toggleGrid());
        
        const stylusFolder = this.gui.addFolder('Stylus Settings');
        stylusFolder.add(this.settings, 'stylusSize', 1, 10, 0.1).name('Stylus Size');
        stylusFolder.add(this.settings, 'stylusSoftness', 0, 1, 0.05).name('Stylus Softness');
        stylusFolder.add(this.settings, 'stylusShape', ['circle', 'square']).name('Stylus Shape');
        
        const displayFolder = this.gui.addFolder('Display Settings');
        displayFolder.add(this.settings, 'invertColors').name('White on Black').onChange(() => this.toggleColorsWithClear());
        displayFolder.add(this.settings, 'mode', ['draw', 'display']).name('Mode').onChange(() => this.toggleMode());
        
        const serverFolder = this.gui.addFolder('Server Settings');
        serverFolder.add(this.settings, 'serverEnabled').name('Server Enabled').onChange(() => this.toggleServer());
        serverFolder.add(this.settings, 'serverUrl').name('Server URL');
        
        const actionsFolder = this.gui.addFolder('Actions');
        actionsFolder.add(this, 'clearGrid').name('Clear Grid');
        actionsFolder.add(this, 'showGridData').name('Show Data');
        actionsFolder.add(this, 'exportGridData').name('Export Data');
        actionsFolder.add(this, 'loadExampleData').name('Load Example');
        actionsFolder.add(this, 'loadFromServer').name('Load from Server');
        actionsFolder.add(this, 'saveToServer').name('Save to Server');
        
        gridFolder.open();
        stylusFolder.open();
        displayFolder.open();
        serverFolder.open();
        actionsFolder.open();
    }
    
    initializeEventListeners() {
        document.getElementById('clearGrid').addEventListener('click', () => this.clearGrid());
        document.getElementById('showData').addEventListener('click', () => this.showGridData());
        document.getElementById('exportData').addEventListener('click', () => this.exportGridData());
        document.getElementById('toggleColors').addEventListener('click', () => this.toggleColorsWithClear());
        document.getElementById('toggleGrid').addEventListener('click', () => this.toggleGridLines());
        document.getElementById('loadData').addEventListener('click', () => this.promptLoadData());
        document.getElementById('serverLoad').addEventListener('click', () => this.loadFromServer());
        document.getElementById('serverSave').addEventListener('click', () => this.saveToServer());
        document.getElementById('serverClear').addEventListener('click', () => this.clearServer());
    }
    
    createGrid() { 
        this.gridRenderer.createGrid(); 
    }
    
    applyStylus(centerIndex) { 
        this.stylusEngine.applyStylus(centerIndex); 
    }
    
    updateCellAppearance(index, value) { 
        this.gridRenderer.updateCellAppearance(index, value); 
    }
    
    clearGrid() { 
        this.dataManager.clearGrid(); 
    }
    
    displayData(dataArray) { 
        return this.dataManager.displayData(dataArray); 
    }
    
    loadFromServer() { 
        this.serverClient.loadFromServer(); 
    }
    
    saveToServer() { 
        this.serverClient.saveToServer(); 
    }
    
    clearServer() { 
        this.serverClient.clearServer(); 
    }
    
    exportGridData() { 
        this.dataManager.exportGridData(); 
    }
    
    showGridData() { 
        this.dataManager.showGridData(); 
    }
    
    loadExampleData() { 
        this.dataManager.loadExampleData(); 
    }
    
    promptLoadData() { 
        this.dataManager.promptLoadData(); 
    }
    
    toggleColorsWithClear() {
        for (let i = 0; i < this.gridData.length; i++) {
            this.gridData[i] = 255 - this.gridData[i];
        }
        
        this.settings.invertColors = !this.settings.invertColors;
        
        const grid = document.querySelector('.grid');
        if (grid) {
            if (!this.settings.showGrid) {
                grid.style.gap = '0px';
                grid.style.backgroundColor = this.settings.invertColors ? 'black' : '#ccc';
            } else {
                grid.style.gap = '1px';
                grid.style.backgroundColor = '#ccc';
            }
        }
        
        for (let i = 0; i < this.gridData.length; i++) {
            this.updateCellAppearance(i, this.gridData[i]);
        }
        
        this.updateInfo();
    }
    
    toggleMode() {
        this.createGrid();
        this.updateInfo();
    }
    
    toggleGrid() {
        const grid = document.querySelector('.grid');
        if (grid) {
            if (!this.settings.showGrid) {
                grid.style.gap = '0px';
                grid.style.backgroundColor = this.settings.invertColors ? 'black' : '#ccc';
            } else {
                grid.style.gap = '1px';
                grid.style.backgroundColor = '#ccc';
            }
        }
    }
    
    toggleGridLines() {
        this.settings.showGrid = !this.settings.showGrid;
        this.toggleGrid();
        this.updateInfo();
    }
    
    toggleServer() {
        if (this.settings.serverEnabled) {
            this.serverClient.testConnection();
        }
    }
    
    updateInfo(additionalInfo = '') {
        const info = document.getElementById('info');
        const mode = this.settings.invertColors ? 'White on Black' : 'Black on White';
        const gridStatus = this.settings.showGrid ? 'Visible' : 'Hidden';
        const interactionMode = this.settings.mode === 'draw' ? 'Drawing' : 'Display';
        const serverStatus = this.settings.serverEnabled ? 'Connected' : 'Disabled';
        
        let infoText = `Grid: ${this.settings.gridSize}×${this.settings.gridSize} | Mode: ${mode} | Grid: ${gridStatus} | Interaction: ${interactionMode} | Server: ${serverStatus}`;
        
        if (additionalInfo) {
            infoText += ` | ${additionalInfo}`;
        }
        
        info.innerHTML = infoText;
    }
    
    getGridData2D() {
        const grid2D = [];
        for (let i = 0; i < this.settings.gridSize; i++) {
            const row = [];
            for (let j = 0; j < this.settings.gridSize; j++) {
                row.push(this.gridData[i * this.settings.gridSize + j]);
            }
            grid2D.push(row);
        }
        return grid2D;
    }
    
    getGridDataFlat() {
        return [...this.gridData];
    }
}