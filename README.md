### 📈 Forecasting de Conversiones por Dimensiones Personalizadas

#### 🎯 El Contexto del Problema 
El equipo de marketing digital busca predecir el impacto de diferentes modelos promocionales en el cumplimiento de su tasa de conversión. El objetivo es transformar las métricas históricas en un modelo de pronóstico capaz de anticipar el rendimiento de las campañas según sus dimensiones operativas, evitando la asignación intuitiva de presupuestos en canales digitales.

---

#### 💡 Hallazgos Clave de la Investigación (Causalidades no Temporales)
El análisis estadístico de las series temporales reveló una estructura atípica en el comportamiento de las conversiones:
- **Tendencia Nula y Estacionariedad Evidente:** Los datos muestran una fluctuación constante alrededor de una media fija, sin una dirección de crecimiento o decrecimiento a lo largo del tiempo.
- **Influencia del Contexto sobre Patrones Estacionales:** Los correlogramas presentaron barras rezagadas muy cortas que oscilan cerca del cero sin un orden claro. Esto confirma la ausencia de ciclos calendarios tradicionales. El volumen de conversión se comporta como eventos independientes discontinuos, impulsados estrictamente por la relevancia del mensaje y el contexto de la oferta.

---

#### 🛠️ Enfoque Técnico y Modelado
Dada la discontinuidad y la falta de estacionalidad temporal clásica, se descartaron los modelos cronológicos tradicionales y se optó por un enfoque basado en aprendizaje supervisado:
- **Ingeniería de Características (Lags y Encoding):** Se crearon variables de rezago (lags) para capturar la inercia operativa inmediata y se aplicó one-hot encoding sobre dimensiones personalizadas como el tipo de campaña.
- **Modelado de Random Forest:** Se entrenó un modelo de bosque aleatorio para aprender la relación entre la configuración estratégica de estimulación comercial, su eficiencia previa y el volumen de conversión resultante.
- **Segmentación por Dimensiones Operativas:** El enfoque permitió que el pronóstico dependiera de las variables estratégicas inyectadas, adaptando el algoritmo a la naturaleza irregular de los eventos promocionales.

---

#### 🚀 Solución Analítica: Simulador de Campañas y Auditoría Digital
El resultado final es una interfaz que no solo automatiza las proyecciones comerciales sino que actúa como un detector de anomalías sistémicas para las diferentes estimulaciones digitales:
- **Explicabilidad del Rendimiento:** Un gráfico horizontal de barras que desglosa el peso e impacto que tiene cada dimensión y variables clave de inercia.
- **Proyección de Dimensiones:** Permite simular escenarios analíticos modificando la variable de contexto seleccionada para evaluar la sensibilidad del retorno antes de ejecutar la inversión real.

---

#### 🎯 Recomendación Estratégica
La minería de datos reveló un comportamiento crítico: un retorno de inversión plano (5%), una tasa de conversión estática (0.08%) y un costo por clic uniforme ($22.7) en la totalidad de las variables analizadas.Lejos de restar valor al Data Product, esta homogeneidad matemática constituye el hallazgo más relevante del proyecto: **diagnostica un sesgo de linealidad sistémica en el origen de los datos**.
Los registros históricos carecen de la variabilidad orgánica del mercado. 
La integración de este simulador interactivo actúa como un espejo analítico de auditoría; demuestra con precisión matemática que el ecosistema actual de datos está viciado o que la empresa opera bajo una distribución empírica plana que anestesia el rendimiento. 
La solución no es optimizar el algoritmo actual, **sino detener la toma de decisiones intuitiva y ejecutar una reestructuración urgente en la captura de métricas y gobierno de datos** para reconstruir la elasticidad del negocio antes de lanzar la próxima estrategia comercial.
