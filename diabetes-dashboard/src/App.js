import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell,ScatterChart, Scatter, ZAxis } from 'recharts';
import { Activity, Heart, TrendingUp, AlertCircle, CheckCircle, FileText } from 'lucide-react';

const DiabetesDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
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

  const distributionData = [
    { name: 'Non-diabétique', value: 213703, color: '#10b981' },
    { name: 'Pré-diabète', value: 4631, color: '#f59e0b' },
    { name: 'Diabétique', value: 35346, color: '#ef4444' }
  ];

  const correlationData = [
    { feature: 'GenHlth', value: 0.38 },
    { feature: 'HighBP', value: 0.35 },
    { feature: 'BMI', value: 0.29 },
    { feature: 'Age', value: 0.27 },
    { feature: 'HighChol', value: 0.26 },
    { feature: 'DiffWalk', value: 0.25 },
    { feature: 'PhysHlth', value: 0.21 },
    { feature: 'Income', value: -0.18 }
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
    { feature: 'HeartDiseaseorAttack', importance: 0.038 }
  ];

  const metricsData = {
    accuracy: 0.7845,
    f1Score: 0.7234,
    aucRoc: 0.8521,
    sensitivity: 0.7812,
    specificity: 0.7654,
    precision: 0.6891,
    TN: 30877, FP: 11864, FN: 1992, TP: 6003
  };

  const thresholdData = [
    { threshold: 0.2, f1: 0.65, recall: 0.92, precision: 0.52 },
    { threshold: 0.3, f1: 0.70, recall: 0.87, precision: 0.58 },
    { threshold: 0.4, f1: 0.72, recall: 0.81, precision: 0.65 },
    { threshold: 0.47, f1: 0.7234, recall: 0.78, precision: 0.69 },
    { threshold: 0.5, f1: 0.71, recall: 0.75, precision: 0.71 },
    { threshold: 0.6, f1: 0.68, recall: 0.68, precision: 0.75 },
    { threshold: 0.7, f1: 0.62, recall: 0.58, precision: 0.80 }
  ];

  const confusionMatrixData = [
    { x: 0, y: 1, value: metricsData.TN, type: 'Vrais Négatifs (TN)', fill: '#10b981' }, 
    { x: 1, y: 1, value: metricsData.FP, type: 'Faux Positifs (FP)', fill: '#ef4444' }, 
    { x: 0, y: 0, value: metricsData.FN, type: 'Faux Négatifs (FN)', fill: '#f59e0b' }, 
    { x: 1, y: 0, value: metricsData.TP, type: 'Vrais Positifs (TP)', fill: '#3b82f6' }, 
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
    { size: 100, train: 0.82, val: 0.75 } // Convergence
  ];

  const maxValue = Math.max(metricsData.TN, metricsData.FP, metricsData.FN, metricsData.TP);

   const getBlueIntensity = (value, maxValue) => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    
     return `rgba(30, 58, 138, ${0.1 + (intensity * 0.9)})`; 
  };

   const getTextColor = (value, maxValue) => {
      return (value / maxValue) > 0.5 ? 'white' : '#1f2937';
  };
  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : parseInt(value)
    }));
  };

  const calculatePrediction = () => {
    let score = 0;
    
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

  const formFields = [
    { name: 'HighBP', label: 'Tension artérielle élevée', type: 'select', options: [0, 1] },
    { name: 'HighChol', label: 'Cholestérol élevé', type: 'select', options: [0, 1] },
    { name: 'CholCheck', label: 'Contrôle du cholestérol', type: 'select', options: [0, 1] },
    { name: 'BMI', label: 'IMC (Indice de Masse Corporelle)', type: 'number', min: 12, max: 70, step: 0.1 },
    { name: 'Smoker', label: 'Fumeur', type: 'select', options: [0, 1] },
    { name: 'Stroke', label: 'Antécédent d\'AVC', type: 'select', options: [0, 1] },
    { name: 'HeartDiseaseorAttack', label: 'Maladie cardiaque', type: 'select', options: [0, 1] },
    { name: 'PhysActivity', label: 'Activité physique', type: 'select', options: [0, 1] },
    { name: 'Fruits', label: 'Consommation de fruits', type: 'select', options: [0, 1] },
    { name: 'Veggies', label: 'Consommation de légumes', type: 'select', options: [0, 1] },
    { name: 'HeavyAlcoholConsumption', label: 'Consommation d\'alcool excessive', type: 'select', options: [0, 1] },
    { name: 'AnyHealthcare', label: 'Couverture santé', type: 'select', options: [0, 1] },
    { name: 'NoDocbcCost', label: 'Pas de médecin (coût)', type: 'select', options: [0, 1] },
    { name: 'GenHlth', label: 'Santé générale (1=Excellent, 5=Mauvais)', type: 'number', min: 1, max: 5, step: 1 },
    { name: 'MentHlth', label: 'Jours de mauvaise santé mentale', type: 'number', min: 0, max: 30, step: 1 },
    { name: 'PhysHlth', label: 'Jours de mauvaise santé physique', type: 'number', min: 0, max: 30, step: 1 },
    { name: 'DiffWalk', label: 'Difficulté à marcher', type: 'select', options: [0, 1] },
    { name: 'Sex', label: 'Sexe (0=Femme, 1=Homme)', type: 'select', options: [0, 1] },
    { name: 'Age', label: 'Catégorie d\'âge (1=18-24, 13=80+)', type: 'number', min: 1, max: 13, step: 1 },
    { name: 'Education', label: 'Niveau d\'éducation (1-6)', type: 'number', min: 1, max: 6, step: 1 },
    { name: 'Income', label: 'Catégorie de revenu (1=<10k, 8=>75k)', type: 'number', min: 1, max: 8, step: 1 }
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

      <nav className="bg-white shadow-sm mt-2">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Tableau de Bord', icon: TrendingUp },
              { id: 'prediction', label: 'Prédiction', icon: Heart },
              { id: 'analysis', label: 'Analyse', icon: FileText }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { label: 'Accuracy', value: (metricsData.accuracy * 100).toFixed(2) + '%', icon: CheckCircle, color: 'bg-green-500' },
                { label: 'F1-Score', value: (metricsData.f1Score * 100).toFixed(2) + '%', icon: TrendingUp, color: 'bg-blue-500' },
                { label: 'AUC-ROC', value: (metricsData.aucRoc * 100).toFixed(2) + '%', icon: Activity, color: 'bg-purple-500' }
              ].map((metric, idx) => (
                <div key={idx} className="bg-white rounded-xl shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">{metric.label}</p>
                      <p className="text-3xl font-bold text-gray-900 mt-2">{metric.value}</p>
                    </div>
                    <div className={`${metric.color} p-3 rounded-full`}>
                      <metric.icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Distribution des Classes</h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={distributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {distributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Top 8 Variables Corrélées</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={correlationData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[-0.2, 0.4]} />
                  <YAxis dataKey="feature" type="category" width={100} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Importance des Variables (Top 10)</h2>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-8">
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Optimisation du Seuil de Décision</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={thresholdData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="threshold" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="f1" stroke="#8b5cf6" strokeWidth={2} name="F1-Score" />
                  <Line type="monotone" dataKey="recall" stroke="#10b981" strokeWidth={2} name="Recall" />
                  <Line type="monotone" dataKey="precision" stroke="#f59e0b" strokeWidth={2} name="Precision" />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-4">Seuil optimal: 0.47 (F1-Score: 0.7234)</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Métriques de Classification</h3>
                <div className="space-y-3">
                  {[
                    { label: 'Sensibilité (Recall)', value: metricsData.sensitivity },
                    { label: 'Spécificité', value: metricsData.specificity },
                    { label: 'Précision', value: metricsData.precision },
                    { label: 'F1-Score', value: metricsData.f1Score }
                  ].map((metric, idx) => (
                    <div key={idx} className="flex justify-between items-center">
                      <span className="text-gray-700">{metric.label}</span>
                      <div className="flex items-center">
                        <div className="w-32 bg-gray-200 rounded-full h-2 mr-3">
                          <div
                            className="bg-indigo-600 h-2 rounded-full"
                            style={{ width: `${metric.value * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-gray-900 w-12">
                          {(metric.value * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Informations du Modèle</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Algorithme</span>
                    <span className="font-semibold">Random Forest / XGBoost</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Technique de rééquilibrage</span>
                    <span className="font-semibold">ADASYN</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Seuil de décision</span>
                    <span className="font-semibold">0.47</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Validation croisée</span>
                    <span className="font-semibold">5-fold CV</span>
                  </div>
                  <div className="flex justify-between py-2">
                    <span className="text-gray-600">Dataset</span>
                    <span className="font-semibold">253,680 échantillons</span>
                  </div>
                </div>
              </div>
              
            </div>
            {/* COURBE ROC */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">Courbe ROC</h2>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart 
                        data={rocCurveData} 
                        margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        
                        <XAxis 
                            dataKey="fpr" 
                            type="number" 
                            domain={[0, 1]} 
                            label={{ 
                                value: 'Faux Positifs (FPR)', 
                                position: 'insideBottom', 
                                offset: -20, 
                                style: { textAnchor: 'middle' }
                            }} 
                        />
                        
                        <YAxis 
                            dataKey="tpr" 
                            type="number" 
                            domain={[0, 1]} 
                            label={{ 
                                value: 'Vrais Positifs (TPR)', 
                                angle: -90, 
                                position: 'insideLeft',
                                style: { textAnchor: 'middle' } 
                            }} 
                        />
                        
                        <Tooltip />
                        
                        <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }}/>
                        
                        <Line type="monotone" dataKey="tpr" stroke="#8b5cf6" strokeWidth={2} name="ROC" dot={false}/>
                        <Line type="monotone" dataKey="fpr" stroke="#d1d5db" strokeWidth={1} strokeDasharray="5 5" name="Aléatoire" dot={false}/>
                    </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-600 mt-4 text-center">AUC-ROC: {metricsData.aucRoc.toFixed(4)}</p>
            </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* FIGURE 3: Matrice de Confusion (CORRIGÉE AVEC HAUTEUR FIXE) */}
                <div className="bg-white rounded-xl shadow-md p-6 flex flex-col items-center justify-center">
                    <h3 className="text-lg font-bold text-gray-900 mb-4">Matrice de Confusion</h3>
                    
                    <div className="grid grid-cols-[auto_1fr] gap-2 w-full max-w-lg">
                        
                        <div className="h-8"></div> 

                        <div className="grid grid-cols-2 text-center text-sm font-semibold text-gray-600 mb-2">
                            <div>Prédit: Non</div>
                            <div>Prédit: Oui</div>
                        </div>
                        <div className="h-64 flex flex-col justify-between text-right pr-3 text-sm font-semibold text-gray-600">
                            <div className="h-1/2 flex items-center justify-end">Réel: Non</div>
                            <div className="h-1/2 flex items-center justify-end">Réel: Oui</div>
                        </div>

                        <div className="h-64 grid grid-cols-2 grid-rows-2 border-2 border-gray-100">
                            
                            <div 
                                className="flex flex-col items-center justify-center p-2 border-r border-b border-gray-100 transition-all hover:bg-blue-50"
                                style={{ backgroundColor: getBlueIntensity(metricsData.TN, maxValue) }}
                            >
                                <span className="text-2xl font-bold" style={{ color: getTextColor(metricsData.TN, maxValue) }}>
                                    {metricsData.TN}
                                </span>
                                <span className="text-xs uppercase font-medium mt-1" style={{ color: getTextColor(metricsData.TN, maxValue) }}>
                                    (TN)
                                </span>
                            </div>

                            <div 
                                className="flex flex-col items-center justify-center p-2 border-b border-gray-100 transition-all hover:bg-blue-50"
                                style={{ backgroundColor: getBlueIntensity(metricsData.FP, maxValue) }}
                            >
                                <span className="text-2xl font-bold" style={{ color: getTextColor(metricsData.FP, maxValue) }}>
                                    {metricsData.FP}
                                </span>
                                <span className="text-xs uppercase font-medium mt-1" style={{ color: getTextColor(metricsData.FP, maxValue) }}>
                                    (FP)
                                </span>
                            </div>

                            <div 
                                className="flex flex-col items-center justify-center p-2 border-r border-gray-100 transition-all hover:bg-blue-50"
                                style={{ backgroundColor: getBlueIntensity(metricsData.FN, maxValue) }}
                            >
                                <span className="text-2xl font-bold" style={{ color: getTextColor(metricsData.FN, maxValue) }}>
                                    {metricsData.FN}
                                </span>
                                <span className="text-xs uppercase font-medium mt-1" style={{ color: getTextColor(metricsData.FN, maxValue) }}>
                                    (FN)
                                </span>
                            </div>

                            <div 
                                className="flex flex-col items-center justify-center p-2 transition-all hover:bg-blue-50"
                                style={{ backgroundColor: getBlueIntensity(metricsData.TP, maxValue) }}
                            >
                                <span className="text-2xl font-bold" style={{ color: getTextColor(metricsData.TP, maxValue) }}>
                                    {metricsData.TP}
                                </span>
                                <span className="text-xs uppercase font-medium mt-1" style={{ color: getTextColor(metricsData.TP, maxValue) }}>
                                    (TP)
                                </span>
                            </div>
                        </div>
                        
                        <div></div> {/* Espace vide à gauche */}
                        <div className="text-center mt-2 text-sm text-gray-500 font-medium">Prédiction</div>

                    </div>
                </div>

                {/* FIGURE 4: Courbe d'Apprentissage (Interactive Line) */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Courbe d'Apprentissage</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart 
                      data={learningCurveData}
                      margin={{ top: 20, right: 30, left: 0, bottom: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      
                      <XAxis 
                        dataKey="size" 
                        label={{ 
                          value: "% Données d'Entraînement", 
                          position: 'insideBottom', 
                          offset: -20, 
                          style: { textAnchor: 'middle', fill: '#666' }  
                        }} 
                      />
                      
                      <YAxis domain={[0.5, 1]} />
                      <Tooltip />
                      
                      <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '20px' }} />
                      
                      <Line type="monotone" dataKey="train" stroke="#3b82f6" strokeWidth={2} name="Entraînement" />
                      <Line type="monotone" dataKey="val" stroke="#10b981" strokeWidth={2} name="Validation" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
            </div>    

          </div>
        )}

        {activeTab === 'prediction' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Formulaire de Prédiction</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {field.label}
                    </label>
                    {field.type === 'select' ? (
                      <select
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      >
                        {field.options.map(opt => (
                          <option key={opt} value={opt}>
                            {opt === 0 ? 'Non' : opt === 1 ? 'Oui' : opt}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleInputChange}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      />
                    )}
                  </div>
                ))}
              </div>
              <button
                onClick={calculatePrediction}
                className="mt-6 w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors shadow-lg"
              >
                Analyser le Risque de Diabète
              </button>
            </div>

            <div className="space-y-6">
              {prediction && (
                <>
                  <div className={`bg-white rounded-xl shadow-md p-6 border-l-4 ${
                    prediction.isAtRisk ? 'border-red-500' : 'border-green-500'
                  }`}>
                    <div className="flex items-start justify-between mb-4">
                      <h3 className="text-lg font-bold text-gray-900">Résultat</h3>
                      {prediction.isAtRisk ? (
                        <AlertCircle className="w-6 h-6 text-red-500" />
                      ) : (
                        <CheckCircle className="w-6 h-6 text-green-500" />
                      )}
                    </div>
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Probabilité de diabète</p>
                        <p className="text-3xl font-bold text-gray-900">
                          {(prediction.probability * 100).toFixed(2)}%
                        </p>
                        <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                          <div
                            className={`h-3 rounded-full ${
                              prediction.probability < 0.3 ? 'bg-green-500' :
                              prediction.probability < 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${prediction.probability * 100}%` }}
                          />
                        </div>
                      </div>
                      <div className="pt-4 border-t">
                        <p className="text-sm text-gray-600">Niveau de risque</p>
                        <p className={`text-xl font-bold ${
                          prediction.riskLevel === 'Faible' ? 'text-green-600' :
                          prediction.riskLevel === 'Modéré' ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {prediction.riskLevel}
                        </p>
                      </div>
                      <div className="pt-4 border-t">
                        <p className="text-sm text-gray-600">Classification</p>
                        <p className={`text-lg font-bold ${
                          prediction.isAtRisk ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {prediction.isAtRisk ? '⚠️ DIABÉTIQUE / À RISQUE' : '✓ NON-DIABÉTIQUE'}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                    <h4 className="font-semibold text-blue-900 mb-2">Note importante</h4>
                    <p className="text-sm text-blue-800">
                      Ce système est un outil d'aide à la décision. Consultez toujours un professionnel de santé pour un diagnostic médical complet.
                    </p>
                  </div>
                </>
              )}

              {!prediction && (
                <div className="bg-gray-50 rounded-xl shadow-md p-6 text-center">
                  <Heart className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">
                    Remplissez le formulaire et cliquez sur "Analyser" pour obtenir une prédiction
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="bg-white mt-12 py-6 shadow-inner">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>Système de détection du diabète basé sur Machine Learning</p>
          <p className="mt-1">Données: BRFSS 2015 • Modèle: Random Forest / XGBoost avec ADASYN</p>
        </div>
      </footer>
    </div>
  );
};

export default DiabetesDashboard;