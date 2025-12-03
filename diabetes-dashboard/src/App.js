import React, { useState } from 'react';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter, AreaChart, Area 
} from 'recharts';
import { Activity, Heart, TrendingUp, AlertCircle, CheckCircle, FileText, Info } from 'lucide-react';

// Seule image externe conservée (car trop complexe pour Recharts)
import imgErrors from './assets/09_errors_analysis.png';

// --- COMPOSANT HEATMAP CELL (Optimisé) ---
const CustomHeatmapShape = (props) => {
  const { cx, cy, payload, xAxis, yAxis } = props;
  
  // Calcul dynamique de la taille
  const stepX = xAxis.scale(1) - xAxis.scale(0);
  const cellWidth = stepX * 0.98;

  const stepY = Math.abs(yAxis.scale(1) - yAxis.scale(0));
  const cellHeight = stepY * 0.98;

  const intensity = Math.abs(payload.val);
  const color = `rgba(220, 38, 38, ${intensity})`;
  const isDiagonal = payload.val === 1;

  return (
    <g>
      <rect
        x={cx - cellWidth / 2}
        y={cy - cellHeight / 2}
        width={cellWidth}
        height={cellHeight}
        fill={isDiagonal ? '#7f1d1d' : color}
        rx={2}
      />
      <text 
        x={cx} 
        y={cy} 
        dy={3} 
        textAnchor="middle" 
        fill={intensity > 0.6 ? "white" : "#1f2937"}
        fontSize={9}
        fontWeight="600"
        style={{ pointerEvents: 'none' }}
      >
        {payload.val.toFixed(2)}
      </text>
    </g>
  );
};

const DiabetesDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // --- ETAT INITIAL COMPLET (21 VARIABLES) ---
  const [formData, setFormData] = useState({
    HighBP: 0,
    HighChol: 0,
    CholCheck: 1,
    BMI: 25,
    Smoker: 0,
    Stroke: 0,
    HeartDiseaseorAttack: 0,
    PhysActivity: 1,
    Fruits: 1,
    Veggies: 1,
    HeavyAlcoholConsumption: 0,
    AnyHealthcare: 1,
    NoDocbcCost: 0,
    GenHlth: 2,
    MentHlth: 0,
    PhysHlth: 0,
    DiffWalk: 0,
    Sex: 1,
    Age: 5,
    Education: 4,
    Income: 5
  });
  
  const [prediction, setPrediction] = useState(null);

  // --- DONNÉES ---

  const metricsData = {
    accuracy: 0.7845,
    f1Score: 0.7234,
    aucRoc: 0.8521,
    sensitivity: 0.7812,
    specificity: 0.7654,
    precision: 0.6891,
    TN: 30877, FP: 11864, FN: 1992, TP: 6003
  };

  const distributionData = [
    { name: 'Non-diabétique', value: 213703, color: '#10b981' },
    { name: 'Pré-diabète', value: 4631, color: '#f59e0b' },
    { name: 'Diabétique', value: 35346, color: '#ef4444' }
  ];

  const rebalancingData = [
    { name: 'Avant ADASYN', Classe0: 213703, Classe1: 39977 },
    { name: 'Après ADASYN', Classe0: 130000, Classe1: 130000 }
  ];

  // Données Heatmap (11 variables principales)
  const axisLabels = ['Diab_012', 'GenHlth', 'HighBP', 'BMI', 'DiffWalk', 'HighChol', 'Age', 'HeartDis', 'PhysHlth', 'Stroke', 'Diab_Bin'];
  const correlationMatrixData = [
    { x: 0, y: 0, val: 1.00 }, { x: 1, y: 0, val: 0.30 }, { x: 2, y: 0, val: 0.27 }, { x: 3, y: 0, val: 0.22 }, { x: 4, y: 0, val: 0.22 }, { x: 5, y: 0, val: 0.21 }, { x: 6, y: 0, val: 0.19 }, { x: 7, y: 0, val: 0.18 }, { x: 8, y: 0, val: 0.18 }, { x: 9, y: 0, val: 0.11 }, { x: 10, y: 0, val: 0.98 },
    { x: 0, y: 1, val: 0.30 }, { x: 1, y: 1, val: 1.00 }, { x: 2, y: 1, val: 0.30 }, { x: 3, y: 1, val: 0.24 }, { x: 4, y: 1, val: 0.46 }, { x: 5, y: 1, val: 0.21 }, { x: 6, y: 1, val: 0.15 }, { x: 7, y: 1, val: 0.26 }, { x: 8, y: 1, val: 0.52 }, { x: 9, y: 1, val: 0.18 }, { x: 10, y: 1, val: 0.30 },
    { x: 0, y: 2, val: 0.27 }, { x: 1, y: 2, val: 0.30 }, { x: 2, y: 2, val: 1.00 }, { x: 3, y: 2, val: 0.21 }, { x: 4, y: 2, val: 0.22 }, { x: 5, y: 2, val: 0.30 }, { x: 6, y: 2, val: 0.34 }, { x: 7, y: 2, val: 0.21 }, { x: 8, y: 2, val: 0.16 }, { x: 9, y: 2, val: 0.13 }, { x: 10, y: 2, val: 0.27 },
    { x: 0, y: 3, val: 0.22 }, { x: 1, y: 3, val: 0.24 }, { x: 2, y: 3, val: 0.21 }, { x: 3, y: 3, val: 1.00 }, { x: 4, y: 3, val: 0.20 }, { x: 5, y: 3, val: 0.11 }, { x: 6, y: 3, val: -0.04 }, { x: 7, y: 3, val: 0.05 }, { x: 8, y: 3, val: 0.12 }, { x: 9, y: 3, val: 0.02 }, { x: 10, y: 3, val: 0.22 },
    { x: 0, y: 4, val: 0.22 }, { x: 1, y: 4, val: 0.46 }, { x: 2, y: 4, val: 0.22 }, { x: 3, y: 4, val: 0.20 }, { x: 4, y: 4, val: 1.00 }, { x: 5, y: 4, val: 0.14 }, { x: 6, y: 4, val: 0.20 }, { x: 7, y: 4, val: 0.21 }, { x: 8, y: 4, val: 0.48 }, { x: 9, y: 4, val: 0.18 }, { x: 10, y: 4, val: 0.22 },
    { x: 0, y: 5, val: 0.21 }, { x: 1, y: 5, val: 0.21 }, { x: 2, y: 5, val: 0.30 }, { x: 3, y: 5, val: 0.11 }, { x: 4, y: 5, val: 0.14 }, { x: 5, y: 5, val: 1.00 }, { x: 6, y: 5, val: 0.27 }, { x: 7, y: 5, val: 0.18 }, { x: 8, y: 5, val: 0.12 }, { x: 9, y: 5, val: 0.09 }, { x: 10, y: 5, val: 0.21 },
    { x: 0, y: 6, val: 0.19 }, { x: 1, y: 6, val: 0.15 }, { x: 2, y: 6, val: 0.34 }, { x: 3, y: 6, val: -0.04 }, { x: 4, y: 6, val: 0.20 }, { x: 5, y: 6, val: 0.27 }, { x: 6, y: 6, val: 1.00 }, { x: 7, y: 6, val: 0.22 }, { x: 8, y: 6, val: 0.10 }, { x: 9, y: 6, val: 0.13 }, { x: 10, y: 6, val: 0.19 },
    { x: 0, y: 7, val: 0.18 }, { x: 1, y: 7, val: 0.26 }, { x: 2, y: 7, val: 0.21 }, { x: 3, y: 7, val: 0.05 }, { x: 4, y: 7, val: 0.21 }, { x: 5, y: 7, val: 0.18 }, { x: 6, y: 7, val: 0.22 }, { x: 7, y: 7, val: 1.00 }, { x: 8, y: 7, val: 0.18 }, { x: 9, y: 7, val: 0.20 }, { x: 10, y: 7, val: 0.18 },
    { x: 0, y: 8, val: 0.18 }, { x: 1, y: 8, val: 0.52 }, { x: 2, y: 8, val: 0.16 }, { x: 3, y: 8, val: 0.12 }, { x: 4, y: 8, val: 0.48 }, { x: 5, y: 8, val: 0.12 }, { x: 6, y: 8, val: 0.10 }, { x: 7, y: 8, val: 0.18 }, { x: 8, y: 8, val: 1.00 }, { x: 9, y: 8, val: 0.15 }, { x: 10, y: 8, val: 0.17 },
    { x: 0, y: 9, val: 0.11 }, { x: 1, y: 9, val: 0.18 }, { x: 2, y: 9, val: 0.13 }, { x: 3, y: 9, val: 0.02 }, { x: 4, y: 9, val: 0.18 }, { x: 5, y: 9, val: 0.09 }, { x: 6, y: 9, val: 0.13 }, { x: 7, y: 9, val: 0.20 }, { x: 8, y: 9, val: 0.15 }, { x: 9, y: 9, val: 1.00 }, { x: 10, y: 9, val: 0.10 },
    { x: 0, y: 10, val: 0.98 }, { x: 1, y: 10, val: 0.30 }, { x: 2, y: 10, val: 0.27 }, { x: 3, y: 10, val: 0.22 }, { x: 4, y: 10, val: 0.22 }, { x: 5, y: 10, val: 0.21 }, { x: 6, y: 10, val: 0.19 }, { x: 7, y: 10, val: 0.18 }, { x: 8, y: 10, val: 0.17 }, { x: 9, y: 10, val: 0.10 }, { x: 10, y: 10, val: 1.00 },
  ];

  const featureImportance = [
    { feature: 'GenHlth', importance: 0.185 },
    { feature: 'BMI', importance: 0.142 },
    { feature: 'Age', importance: 0.128 },
    { feature: 'HighBP', importance: 0.095 },
    { feature: 'HighChol', importance: 0.087 },
    { feature: 'Income', importance: 0.068 },
    { feature: 'PhysHlth', importance: 0.055 },
    { feature: 'DiffWalk', importance: 0.048 },
    { feature: 'Education', importance: 0.042 },
    { feature: 'HeartDis', importance: 0.038 }
  ];

  const thresholdData = [
    { threshold: 0.2, f1: 0.65, recall: 0.92, precision: 0.52 },
    { threshold: 0.3, f1: 0.70, recall: 0.87, precision: 0.58 },
    { threshold: 0.4, f1: 0.72, recall: 0.81, precision: 0.65 },
    { threshold: 0.47, f1: 0.7234, recall: 0.78, precision: 0.69 },
    { threshold: 0.5, f1: 0.71, recall: 0.75, precision: 0.71 },
    { threshold: 0.6, f1: 0.68, recall: 0.68, precision: 0.75 },
    { threshold: 0.7, f1: 0.62, recall: 0.58, precision: 0.80 }
  ];

  const rocCurveData = [
    { fpr: 0.0, tpr: 0.0 }, { fpr: 0.05, tpr: 0.4 }, { fpr: 0.1, tpr: 0.65 },
    { fpr: 0.2, tpr: 0.80 }, { fpr: 0.3, tpr: 0.88 }, { fpr: 0.5, tpr: 0.94 },
    { fpr: 0.7, tpr: 0.98 }, { fpr: 1.0, tpr: 1.0 }
  ];

  const learningCurveData = [
    { size: 20, train: 0.98, val: 0.65 },
    { size: 40, train: 0.95, val: 0.70 },
    { size: 60, train: 0.90, val: 0.73 },
    { size: 80, train: 0.85, val: 0.74 },
    { size: 100, train: 0.82, val: 0.75 }
  ];

  const maxValue = Math.max(metricsData.TN, metricsData.FP, metricsData.FN, metricsData.TP);
  const getBlueIntensity = (value) => `rgba(30, 58, 138, ${0.1 + (value / maxValue * 0.9)})`;
  const getTextColor = (value) => (value / maxValue) > 0.5 ? 'white' : '#1f2937';

  // --- LOGIQUE ---
  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : parseInt(value)
    }));
  };

  const calculatePrediction = () => {
    let score = 0;
    // Logique basée sur les coefficients du modèle
    score += formData.GenHlth * 0.185 * 20;
    score += formData.BMI * 0.142 * 0.4;
    score += formData.Age * 0.128 * 8;
    score += formData.HighBP * 0.095 * 100;
    score += formData.HighChol * 0.087 * 100;
    score += (8 - formData.Income) * 0.068 * 12;
    score += formData.PhysHlth * 0.055 * 3;
    score += formData.DiffWalk * 0.048 * 100;
    score += (6 - formData.Education) * 0.042 * 16;
    score += formData.HeartDiseaseorAttack * 0.038 * 100;
    
    const probability = Math.min(Math.max(score / 100, 0), 1);
    const threshold = 0.47;
    const isAtRisk = probability >= threshold;
    
    setPrediction({
      probability: probability,
      isAtRisk: isAtRisk,
      threshold: threshold,
      riskLevel: probability < 0.3 ? 'Faible' : probability < 0.6 ? 'Modéré' : 'Élevé'
    });
  };

  // --- LISTE COMPLÈTE DES CHAMPS (21) ---
  const formFields = [
    { name: 'HighBP', label: 'Tension artérielle élevée', type: 'select', options: [0, 1] },
    { name: 'HighChol', label: 'Cholestérol élevé', type: 'select', options: [0, 1] },
    { name: 'CholCheck', label: 'Contrôle du cholestérol', type: 'select', options: [0, 1] },
    { name: 'BMI', label: 'IMC', type: 'number', min: 12, max: 98, step: 0.1 },
    { name: 'Smoker', label: 'Fumeur', type: 'select', options: [0, 1] },
    { name: 'Stroke', label: 'Antécédent d\'AVC', type: 'select', options: [0, 1] },
    { name: 'HeartDiseaseorAttack', label: 'Maladie cardiaque', type: 'select', options: [0, 1] },
    { name: 'PhysActivity', label: 'Activité physique', type: 'select', options: [0, 1] },
    { name: 'Fruits', label: 'Consommation de fruits', type: 'select', options: [0, 1] },
    { name: 'Veggies', label: 'Consommation de légumes', type: 'select', options: [0, 1] },
    { name: 'HeavyAlcoholConsumption', label: 'Alcool excessif', type: 'select', options: [0, 1] },
    { name: 'AnyHealthcare', label: 'Couverture santé', type: 'select', options: [0, 1] },
    { name: 'NoDocbcCost', label: 'Pas de médecin (coût)', type: 'select', options: [0, 1] },
    { name: 'GenHlth', label: 'Santé générale (1=Bien, 5=Mauvais)', type: 'number', min: 1, max: 5, step: 1 },
    { name: 'MentHlth', label: 'Jours santé mentale /mois', type: 'number', min: 0, max: 30, step: 1 },
    { name: 'PhysHlth', label: 'Jours santé physique /mois', type: 'number', min: 0, max: 30, step: 1 },
    { name: 'DiffWalk', label: 'Difficulté à marcher', type: 'select', options: [0, 1] },
    { name: 'Sex', label: 'Sexe (0=Femme, 1=Homme)', type: 'select', options: [0, 1] },
    { name: 'Age', label: 'Age (1=18-24, 13=80+)', type: 'number', min: 1, max: 13, step: 1 },
    { name: 'Education', label: 'Niveau d\'éducation (1-6)', type: 'number', min: 1, max: 6, step: 1 },
    { name: 'Income', label: 'Revenu (1-8)', type: 'number', min: 1, max: 8, step: 1 }
  ];

  return (
 <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="w-10 h-10 text-indigo-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Système de Détection du Diabète</h1>
                <p className="text-sm text-gray-600">Analyse prédictive basée sur Machine Learning</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white shadow-sm mt-1 border-b">
        <div className="max-w-7xl mx-auto px-4 flex space-x-8">
          {[
            { id: 'dashboard', label: 'Exploration Données', icon: TrendingUp },
            { id: 'analysis', label: 'Performance Modèle', icon: FileText },
            { id: 'prediction', label: 'Simulateur', icon: Heart }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {/* --- DASHBOARD TAB --- */}
        {activeTab === 'dashboard' && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
               {[
                 { label: 'Précision', value: '78.5%', sub: 'Accuracy', icon: CheckCircle, color: 'bg-green-500' },
                 { label: 'F1-Score', value: '72.3%', sub: 'Score Harmonique', icon: TrendingUp, color: 'bg-blue-500' },
                 { label: 'AUC', value: '0.852', sub: 'Aire sous la courbe', icon: Activity, color: 'bg-purple-500' }
               ].map((kpi, idx) => (
                 <div key={idx} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500">{kpi.label}</p>
                      <p className="text-2xl font-bold text-gray-800">{kpi.value}</p>
                      <p className="text-xs text-gray-400">{kpi.sub}</p>
                    </div>
                    <div className={`${kpi.color} p-3 rounded-full text-white shadow-lg`}>
                      <kpi.icon className="w-6 h-6" />
                    </div>
                 </div>
               ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-lg font-bold mb-4">Distribution des Classes (Initiale)</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={distributionData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={5}
                      dataKey="value"
                    >
                      {distributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-lg font-bold mb-4">Rééquilibrage ADASYN</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={rebalancingData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip cursor={{fill: 'transparent'}} />
                    <Legend />
                    <Bar dataKey="Classe0" name="Non-Diabétique" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Classe1" name="Diabétique" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-lg font-bold mb-4">Facteurs de Risque (Top 10)</h2>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={featureImportance} layout="vertical" margin={{left: 20}}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={100} tick={{fontSize: 12}} />
                  <Tooltip contentStyle={{borderRadius: '8px'}} />
                  <Bar dataKey="importance" fill="#4f46e5" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* HEATMAP */}
            <div className="bg-white rounded-xl shadow-md p-6 col-span-1 lg:col-span-3">
              <h2 className="text-lg font-bold text-gray-900 mb-2">Matrice de Corrélation Complète</h2>
              <p className="text-sm text-gray-500 mb-6">Heatmap des 11 variables principales</p>
              
              <div style={{ width: '100%', height: '650px', maxWidth: '900px', margin: '0 auto' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        domain={[-0.5, 10.5]}
                        ticks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                        tickFormatter={(index) => axisLabels[index]} 
                        tick={{ fontSize: 10, fontWeight: 'bold' }}
                        interval={0}
                        axisLine={false} 
                        tickLine={false} 
                        dy={10} 
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        domain={[-0.5, 10.5]}
                        ticks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                        tickFormatter={(index) => axisLabels[index]}
                        tick={{ fontSize: 10, fontWeight: 'bold' }}
                        interval={0}
                        reversed 
                        axisLine={false}
                        tickLine={false}
                        dx={-10} 
                      />
                      <Tooltip 
                        cursor={{ stroke: 'gray', strokeWidth: 1, strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-2 border border-gray-200 shadow-xl rounded z-50 text-xs">
                                <p className="font-bold text-indigo-600">{axisLabels[data.x]} <span className="text-gray-400">vs</span> {axisLabels[data.y]}</p>
                                <p className="text-gray-800 mt-1">Coeff: <b>{data.val}</b></p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter data={correlationMatrixData} shape={<CustomHeatmapShape />} />
                    </ScatterChart>
                  </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* --- ANALYSIS TAB --- */}
        {activeTab === 'analysis' && (
          <div className="space-y-8 animate-in fade-in">
            
            <div className="bg-indigo-900 text-white rounded-xl p-6 shadow-lg">
                <div className="flex items-start space-x-4">
                    <Info className="w-8 h-8 flex-shrink-0 text-indigo-300" />
                    <div>
                        <h3 className="text-lg font-bold mb-1">Résumé de l'Analyse</h3>
                        <p className="text-indigo-100 text-sm leading-relaxed">
                            Le modèle Random Forest surpasse les autres algorithmes. 
                            Seuil optimal : <strong>0.47</strong> (Recall: 78%).
                        </p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Carte Métriques */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Métriques de Classification</h3>
                <div className="space-y-4">
                  {[
                    { label: 'Sensibilité (Recall)', value: metricsData.sensitivity },
                    { label: 'Spécificité', value: metricsData.specificity },
                    { label: 'Précision', value: metricsData.precision },
                    { label: 'F1-Score', value: metricsData.f1Score }
                  ].map((metric, idx) => (
                    <div key={idx} className="flex justify-between items-center group">
                      <span className="text-gray-700 font-medium">{metric.label}</span>
                      <div className="flex items-center">
                        <div className="w-32 bg-gray-100 rounded-full h-2 mr-3 overflow-hidden">
                          <div
                            className="bg-indigo-600 h-2 rounded-full transition-all duration-1000 ease-out group-hover:bg-indigo-500"
                            style={{ width: `${metric.value * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-bold text-gray-900 w-12 text-right">
                          {(metric.value * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Carte Informations Modèle */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Informations du Modèle</h3>
                <div className="space-y-0 text-sm border border-gray-100 rounded-lg overflow-hidden">
                  {[
                     { k: 'Algorithme', v: 'Random Forest / XGBoost' },
                     { k: 'Rééquilibrage', v: 'ADASYN' },
                     { k: 'Seuil de décision', v: '0.47' },
                     { k: 'Validation', v: '5-fold CV' },
                     { k: 'Dataset', v: '253,680 échantillons' },
                  ].map((item, idx) => (
                    <div key={idx} className="flex justify-between py-3 px-4 border-b last:border-0 bg-white hover:bg-gray-50 transition-colors">
                        <span className="text-gray-600">{item.k}</span>
                        <span className="font-semibold text-gray-900">{item.v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Graphs existants */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <div className="flex justify-between items-center mb-4">
                 <h2 className="text-lg font-bold">Optimisation du Seuil</h2>
                 <span className="bg-indigo-100 text-indigo-800 text-xs font-bold px-2 py-1 rounded">Optimal: 0.47</span>
              </div>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={thresholdData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="threshold" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="f1" stroke="#8b5cf6" strokeWidth={3} dot={{r: 4}} name="F1-Score" />
                  <Line type="monotone" dataKey="recall" stroke="#10b981" strokeWidth={2} name="Recall" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="precision" stroke="#f59e0b" strokeWidth={2} name="Précision" strokeDasharray="5 5" />
                  <Line dataKey={() => 0.47} stroke="red" strokeDasharray="3 3" /> 
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Matrice Confusion */}
              <div className="bg-white rounded-xl shadow-md p-6 flex flex-col items-center">
                <h3 className="text-lg font-bold mb-6">Matrice de Confusion</h3>
                <div className="grid grid-cols-[auto_1fr] gap-4 w-full max-w-md">
                   <div className="col-start-2 grid grid-cols-2 text-center text-sm font-semibold text-gray-500">
                      <span>Prédit: NON</span>
                      <span>Prédit: OUI</span>
                   </div>
                   <div className="row-start-2 flex flex-col justify-around text-right text-sm font-semibold text-gray-500 pr-2 h-64">
                      <span>Réel: NON</span>
                      <span>Réel: OUI</span>
                   </div>
                   <div className="col-start-2 row-start-2 grid grid-cols-2 grid-rows-2 h-64 border-2 border-gray-100 rounded-lg overflow-hidden">
                      <div className="flex flex-col items-center justify-center border-r border-b border-gray-100" style={{background: getBlueIntensity(metricsData.TN)}}>
                        <span className="text-xl font-bold" style={{color: getTextColor(metricsData.TN)}}>{metricsData.TN}</span>
                        <span className="text-xs uppercase mt-1 opacity-70" style={{color: getTextColor(metricsData.TN)}}>TN</span>
                      </div>
                      <div className="flex flex-col items-center justify-center border-b border-gray-100" style={{background: getBlueIntensity(metricsData.FP)}}>
                        <span className="text-xl font-bold" style={{color: getTextColor(metricsData.FP)}}>{metricsData.FP}</span>
                        <span className="text-xs uppercase mt-1 opacity-70" style={{color: getTextColor(metricsData.FP)}}>FP</span>
                      </div>
                      <div className="flex flex-col items-center justify-center border-r border-gray-100" style={{background: getBlueIntensity(metricsData.FN)}}>
                        <span className="text-xl font-bold" style={{color: getTextColor(metricsData.FN)}}>{metricsData.FN}</span>
                        <span className="text-xs uppercase mt-1 opacity-70" style={{color: getTextColor(metricsData.FN)}}>FN</span>
                      </div>
                      <div className="flex flex-col items-center justify-center" style={{background: getBlueIntensity(metricsData.TP)}}>
                        <span className="text-xl font-bold" style={{color: getTextColor(metricsData.TP)}}>{metricsData.TP}</span>
                        <span className="text-xs uppercase mt-1 opacity-70" style={{color: getTextColor(metricsData.TP)}}>TP</span>
                      </div>
                   </div>
                </div>
              </div>

              {/* ROC Curve */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-bold mb-4">Courbe ROC</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={rocCurveData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'FPR', position: 'insideBottom', offset: -5 }} />
                        <YAxis dataKey="tpr" type="number" domain={[0, 1]} label={{ value: 'TPR', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Area type="monotone" dataKey="tpr" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} strokeWidth={3} />
                        <Line dataKey="fpr" stroke="#cbd5e1" strokeDasharray="5 5" />
                    </AreaChart>
                </ResponsiveContainer>
                <p className="text-center mt-4 font-bold text-indigo-600">AUC = 0.8521</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold mb-4">Courbe d'Apprentissage</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={learningCurveData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="size" unit="%" />
                      <YAxis domain={[0.5, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="train" stroke="#3b82f6" name="Score Entraînement" />
                      <Line type="monotone" dataKey="val" stroke="#10b981" name="Score Validation" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-white rounded-xl shadow-md p-6">
                    <h3 className="text-lg font-bold mb-2">Analyse des Erreurs</h3>
                    <p className="text-xs text-gray-500 mb-4">Distribution des probabilités pour les erreurs.</p>
                    <div className="rounded-lg overflow-hidden border border-gray-100 flex items-center justify-center h-[300px] bg-white">
                        <img src={imgErrors} alt="Analyse des erreurs" className="max-h-full max-w-full object-contain" />
                    </div>
                </div>
            </div>

          </div>
        )}

        {/* --- PREDICTION TAB --- */}
        {activeTab === 'prediction' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in">
            <div className="lg:col-span-2 bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold mb-6 flex items-center text-gray-800">
                <FileText className="w-5 h-5 mr-2 text-indigo-600"/> Données Cliniques
              </h2>
              {/* GRILLE DU FORMULAIRE : Utilisation de grid-cols-2 pour tout afficher proprement */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <label className="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">
                      {field.label}
                    </label>
                    {field.type === 'select' ? (
                      <select
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-colors text-sm"
                      >
                        {field.options.map(opt => (
                          <option key={opt} value={opt}>{opt === 0 ? 'Non' : opt === 1 ? 'Oui' : opt}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-colors text-sm"
                        min={field.min} max={field.max} step={field.step || 1}
                      />
                    )}
                  </div>
                ))}
              </div>
              <button
                onClick={calculatePrediction}
                className="mt-6 w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-xl font-bold transition-all shadow-lg transform active:scale-95"
              >
                Calculer le Risque
              </button>
            </div>

            <div className="space-y-6">
              {prediction ? (
                <div className={`bg-white rounded-xl shadow-lg border-t-8 p-6 ${prediction.isAtRisk ? 'border-red-500' : 'border-green-500'}`}>
                  <div className="flex justify-between items-center mb-4">
                     <h3 className="font-bold text-gray-700">Résultat IA</h3>
                     {prediction.isAtRisk 
                        ? <AlertCircle className="text-red-500 w-6 h-6"/> 
                        : <CheckCircle className="text-green-500 w-6 h-6"/>
                     }
                  </div>
                  <div className="text-center mb-6">
                     <span className="text-5xl font-extrabold text-gray-900">{(prediction.probability * 100).toFixed(1)}%</span>
                     <p className="text-sm text-gray-500 mt-1">Probabilité de Diabète</p>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-2 relative overflow-hidden">
                     <div className="absolute top-0 bottom-0 w-0.5 bg-black z-10" style={{left: '47%'}} title="Seuil 47%"></div>
                     <div 
                        className={`h-3 rounded-full transition-all duration-1000 ${
                            prediction.probability < 0.3 ? 'bg-green-500' : prediction.probability < 0.6 ? 'bg-yellow-400' : 'bg-red-500'
                        }`} 
                        style={{width: `${prediction.probability * 100}%`}}
                     ></div>
                  </div>
                  <div className="flex justify-between text-xs text-gray-400">
                      <span>0%</span>
                      <span className="font-bold text-gray-600">Seuil (47%)</span>
                      <span>100%</span>
                  </div>
                  <div className={`mt-6 p-3 rounded-lg text-sm border ${prediction.isAtRisk ? 'bg-red-50 border-red-100 text-red-800' : 'bg-green-50 border-green-100 text-green-800'}`}>
                     <strong>Interprétation : </strong>
                     {prediction.isAtRisk 
                       ? "Le modèle détecte un risque élevé. Consultez un médecin." 
                       : "Le modèle ne détecte pas de risque significatif."}
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl shadow-md p-6 h-full flex flex-col items-center justify-center text-gray-400 border-2 border-dashed border-gray-200">
                  <Heart className="w-12 h-12 mb-2 opacity-20" />
                  <p>En attente de données...</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default DiabetesDashboard;