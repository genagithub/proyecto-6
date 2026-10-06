### 📈 Forecasting de Conversiones 

#### 🎯 El Contexto del Problema 
El equipo de marketing digital se enfrenta al desafío constante de maximizar la tasa de conversión de leads, entendidos como el registro de prospectos calificados en entornos volátiles. Ante la falta de visibilidad del éxito de las pautas, las decisiones tácticas y presupuestarias corren el riesgo de asignarse de forma intuitiva o basándose en correlaciones superficiales que comprometen el retorno de la inversión. El objetivo es transformar el histórico transaccional en un motor de descubrimiento analítico y pronóstico mediante un modelo econométrico con variables exógenas, de manera que se decodifiquen los patrones temporales ocultos e identifiquen el impacto neto de cada dimensión operativa sobre la conversión final.

---

#### 💡 Hallazgos Clave de la Investigación (Causalidades Actuales)
Se desmitificó las correlaciones superficiales del rendimiento publicitario mediante un desglose secuencial, revelando dos dinámicas críticas sobre el comportamiento del mercado:
- **Homogeneidad Estacional Absoluta**: La variación en el volumen de conversiones entre el mejor día de la semana (Lunes) y el peor (Fin de semana) es de apenas un 1.2%, demostrando que la demanda de leads es lineal y el consumidor digital reacciona con la misma consistencia de lunes a domingo.
- **Predominancia de Estímulos Inmediatos (Ruido Operativo)**: Al aislar la estacionalidad semanal, la volatilidad diaria de la serie responde de forma directa a las configuraciones tácticas del día (canales y presupuestos activos), confirmando que el ecosistema no se mueve por inercia temporal pasada, sino por los impactos presupuestarios del presente.

---

#### 🛠️ Enfoque Técnico y Modelado
Para resolver la opacidad analítica de las pautas sin caer en el sobreajuste que provocan los modelos de cajas negras en series de tiempo cortas, se implementó un flujo econométrico paramétrico estructurado:
- **Ingeniería de Variables por Intensidad (Matriz Exógena)**: Se transformaron más de 200,000 registros transaccionales en una serie diaria unificada de 365 días mediante un pivoteo de agregación. Las dimensiones categóricas se ponderaron según el volumen de clics diarios, permitiendo al modelo aislar el tiempo y evaluar el impacto neto de cada estímulo operativo.
- **Parametrización SARIMAX(1, 0, 1) y Parsimonia**: Respaldado por un Test de Dickey-Fuller contundente que confirmó estacionariedad nativa y funciones ACF/PACF de memoria corta, se configuró una estructura autorregresiva de medias móviles ligera. Este enfoque lineal protege el ecosistema contra el overfitting y garantiza que el peso de la predicción recaiga en la combinación de pautas del día.
- **Eficiencia Operativa de Alta Precisión**: El modelo demostró una alta precisión con un Error Porcentual Absoluto Medio (MAPE) de apenas 1.58% y un Error Absoluto Medio (MAE) de solo 378 conversiones diarias. El Coeficiente de Determinación (R^2 = 0.51) valida que el modelo explica más de la mitad de la varianza pura del mercado publicitario, ofreciendo un comportamiento de suavizado controlado óptimo para planeación financiera.

---

#### 🚀 Solución Analítica: Proyección de Conversiones y Medición del Impacto
El resultado final es una interfaz interactiva que funciona como un centro de control ejecutivo que audita el pasado y simula el futuro a través de tres capacidades estratégicas:
- **Auditoría de Atribución Neta (Mix de Medios)**: Traduce los coeficientes estadísticos en un gráfico interactivo de impacto marginal, se revela el peso real de cada variable aislando factores externos y permitiendo identificar de inmediato los verdaderos motores de conversión del negocio.
- **Desmitificación de Canales**: Detecta qué componentes no están moviendo la aguja (como se demostró estadísticamente con el caso de YouTube). Esto permite al equipo redefinir la estrategia de estos canales hacia objetivos de branding o asistencia en el embudo superior, deteniendo la asignación intuitiva de presupuestos de performance.
- **Simulación de Horizontes Futuros Dinámicos**: Estima el volumen de leads totales bajo escenarios estables de pauta, el gráfico de tendencia futura se complementa con tres tarjetas de control financiero (CPA, CVR y CPC Medios Proyectados), permitiendo a la dirección previsualizar si el escenario simulado mantendrá la eficiencia de costos antes de arriesgar capital en el mercado real.

---

#### 📌 Propósito de este Proyecto: Impacto Estratégico
- **Maximización de la Eficiencia de Capital Operativo:** Al fusionar la precisión de un pronóstico con el descubrimiento analítico del mix de medios, el proyecto impacta directamente en las finanzas de la empresa: no busca simplemente gastar más, sino reestructurar de forma inteligente la inversión hacia los vectores de campaña con mayor elasticidad de conversión, lo que se traduce en una estrategia de rentabilidad agresiva que maximiza el Retorno de la Inversión en Marketing (ROMI) y mitiga el Costo de Adquisición de Clientes (CAC), permitiendo a la organización capturar un mayor volumen de prospectos calificados utilizando los mismos recursos financieros disponibles.
