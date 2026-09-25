# Introducción a la Inteligencia Artificial para Bioacústica

Este repositorio contiene los materiales del curso AI for Bioacoustics del 3er Congreso Colombiano de Bioacústica y Ecoacústica. 

Oprime el siguiente enlace para abrir el taller [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jscanass/3CCBE_AI4Bioacoustics/blob/main/Notebook.ipynb)

![logo](https://redecoacustica.org/assets/images/3ccbe_image.png)

## Descripción del taller

El monitoreo acústico pasivo (PAM) genera volúmenes masivos de audio ambiental que hacen inviable el análisis manual. Este taller ofrece una introducción práctica al uso de **Machine Learning** y **Python** para la clasificación automática de vocalizaciones de especies, transformando datos brutos en información ecológica valiosa.

A través de un enfoque práctico y el uso de herramientas de código abierto, los participantes explorarán todo el *pipeline* analítico: desde la lógica detrás de los algoritmos de reconocimiento de patrones y representación de audio (espectrogramas), hasta su aplicación en escenarios de campo reales, evaluando tanto oportunidades como limitaciones metodológicas.

El objetivo principal es dotar a los asistentes de las habilidades técnicas necesarias para gestionar proyectos de monitoreo a gran escala, dominando el ecosistema de herramientas disponibles y la implementación de modelos en datos reales.

---

## Información del evento

* **Evento:** 3er Congreso Colombiano de Bioacústica y Ecoacústica ([3CCBE](https://3ccbe.com/))
* **Fecha:** Lunes (9:30 - 13:00) 28 de septiembre de 2026.
* **Lugar:** Auditorio Félix Restrepo - Pontificia Universidad Javeriana, Bogotá, Colombia

---

## Instructores

* **Juan Sebastián Cañas** – *University College London (UCL)*
* **Daniela Ruiz** – *Microsoft AI for Good / University College London (UCL)*
* **Manuel Alejandro Jaramillo Rodríguez** – *KU Leuven*
* **Maria J. Guerrero** – *Universidad de Antioquia / Rice University*

---

## Contenido Principal

1. **Introducción y Fundamentos:** Audición computacional y principios de la recolección de datos acústicos en ecología.
2. **Visualización de Sonidos:** Análisis mediante formas de onda y espectrogramas (incluyendo el impacto de los parámetros STFT).
3. **Detección Automática de Sonidos:** Uso de modelos pre-entrenados y evaluación de rendimiento (precisión, *recall* e Intersección sobre Unión - IoU).
4. **Identificación y Clasificación de Especies:** Extracción de características (embeddings) mediante *transfer learning* con modelos generales (YamNet, Perch) y entrenamiento de clasificadores (Random Forest).
5. **Conclusiones y Retos:** Discusión sobre *domain shift* y buenas prácticas en ecoacústica computacional.

---

## Estructura del Repositorio

↳ presentacion → contiene pdf con la presentación para la sesión teórica

↳ notebook → cuaderno principal en colab del taller en python para la sesión práctica 

## Material adicional

Si te interesa alguno de los temas que hemos abordado hoy, aquí hay una breve lista de lecturas recomendadas para comenzar a explorar estas ideas con mayor profundidad.

- Ghani et al. 2026. Twelve quick tips for applying deep learning to animal sounds. PLOS Computational Biology. [https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(25)00627-0/fulltext](https://doi.org/10.1371/journal.pcbi.1014604
)

---

## Agradecimientos y créditos

Este curso toma como base el plan de estudios de la asignatura *AI for Environment* perteneciente a la Maestría en *Ecology and Data Science* de University College London (UCL).
🔗 [Repositorio base de UCL - BIOS0032_AI4Environment](https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment)
