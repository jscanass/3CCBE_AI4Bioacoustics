from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from urllib.request import urlretrieve
import hashlib
import json
import warnings
import zipfile

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import Audio, display
from matplotlib.patches import Rectangle
from scipy.io import wavfile
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ZENODO_RECORD = "21829388"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ARCHIVOS_PEQUENOS = {
    "annotations_species.json": "fefae0bd5b9cad4ead154ebf4dfeaf37",
    "species.csv": "ab0dd4b6ecfd526c749b86bd6a5e71cc",
}
ZIP_PROYECTOS = {
    "MAP1": ("MAP1.zip", "499ce381754892ebd8311433fd8bc99c"),
    "PPA1": ("PPA1.zip", "d5fce3e9d743bad4988131b57aa1c9bc"),
    "PPA2": ("PPA2.zip", "42b4d4c7c27c821beab86dd51dc2204a"),
    "PPA3": ("PPA3.zip", "c8e4d32ec0ccac4fd0928ee4dbc91abe"),
    "PPA4": ("PPA4.zip", "aba69c06c5b6d8f749e56d1c85150898"),
}


def ruta_datos_predeterminada():
    try:
        from google.colab import drive
    except ImportError:
        return Path(r"G:\My Drive\Bioacoustic AI - CCBE3\Curso\pteroset_data")

    drive.mount("/content/drive")
    return Path("/content/drive/MyDrive/Bioacoustic AI - CCBE3/Curso/pteroset_data")


def _md5_archivo(ruta, tamano_bloque=1024 * 1024):
    resumen = hashlib.md5()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(tamano_bloque), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def _nombre_archivo(ruta):
    valor = str(ruta)
    return PureWindowsPath(valor).name if "\\" in valor else PurePosixPath(valor).name


@dataclass
class PteroSet:
    raiz: Path
    proyecto: str = "MAP1"
    duracion_segmento: float = 10.0
    descargar_audio: bool = True

    def __post_init__(self):
        self.raiz = Path(self.raiz).expanduser().resolve()
        self.raiz.mkdir(parents=True, exist_ok=True)
        if self.proyecto not in ZIP_PROYECTOS:
            raise ValueError(f"proyecto debe ser uno de {sorted(ZIP_PROYECTOS)}")

        for nombre, md5_esperado in ARCHIVOS_PEQUENOS.items():
            self._descargar_zenodo(nombre, md5_esperado)

        with open(
            self.raiz / "annotations_species.json", encoding="utf-8"
        ) as archivo:
            pteroset = json.load(archivo)

        self.taxonomia = pd.read_csv(self.raiz / "species.csv")
        self.sonidos = pd.DataFrame(pteroset["sounds"]).set_index("id")
        self.anotaciones = pd.DataFrame(pteroset["annotations"])
        self.categorias = pd.DataFrame(pteroset["categories"])
        self.sonidos_proyecto = self.sonidos.loc[
            self.sonidos["project"].eq(self.proyecto)
        ].reset_index()
        if self.sonidos_proyecto.empty:
            raise ValueError(
                f"Las anotaciones no contienen el proyecto {self.proyecto}."
            )

        self.sonidos_proyecto["audio_file"] = self.sonidos_proyecto[
            "file_name_path"
        ].map(lambda ruta: Path(str(ruta).replace("\\", "/")).name)
        self._actualizar_wav()
        if self.descargar_audio:
            self._asegurar_wav()

    @property
    def ruta_etiquetas(self):
        return self.raiz / f"{self.proyecto.lower()}_paso10_etiquetas_5s.csv"

    @property
    def ruta_embeddings(self):
        return self.raiz / f"{self.proyecto.lower()}_paso10_perch2_embeddings_5s.csv"

    @property
    def sonido_id_por_archivo(self):
        return dict(
            zip(
                self.sonidos_proyecto["audio_file"],
                self.sonidos_proyecto["id"],
            )
        )

    def _descargar_zenodo(self, nombre, md5_esperado):
        destino = self.raiz / nombre
        if destino.exists() and _md5_archivo(destino) == md5_esperado:
            return destino
        if destino.exists():
            destino.unlink()

        print(f"Descargando {nombre}...")
        urlretrieve(f"{ZENODO_API}/files/{nombre}/content", destino)
        if _md5_archivo(destino) != md5_esperado:
            destino.unlink(missing_ok=True)
            raise OSError(f"La suma MD5 de {nombre} no coincide con Zenodo.")
        return destino

    def _asegurar_wav(self):
        zip_nombre, zip_md5 = ZIP_PROYECTOS[self.proyecto]
        self._actualizar_wav()
        requeridos = set(self.sonidos_proyecto["audio_file"])
        faltantes = requeridos.difference(self.wav_por_nombre)

        if not faltantes:
            return

        ruta_zip = self._descargar_zenodo(zip_nombre, zip_md5)
        with zipfile.ZipFile(ruta_zip) as comprimido:
            raiz_resuelta = self.raiz.resolve()
            for miembro in comprimido.infolist():
                destino = (self.raiz / miembro.filename).resolve()
                if raiz_resuelta not in destino.parents and destino != raiz_resuelta:
                    raise ValueError(
                        f"Ruta insegura dentro de {zip_nombre}: {miembro.filename}"
                    )
            comprimido.extractall(self.raiz)

        self._actualizar_wav()
        faltantes = requeridos.difference(self.wav_por_nombre)
        if faltantes:
            raise FileNotFoundError(
                f"Faltan {len(faltantes)} WAV de {self.proyecto}. "
                f"Primer ejemplo: {sorted(faltantes)[0]}"
            )

    def _actualizar_wav(self):
        self.wav_por_nombre = {
            ruta.name: ruta.resolve() for ruta in self.raiz.rglob("*.wav")
        }

    def _normalizar_indice(self, tabla):
        archivos = tabla.index.get_level_values("file")
        tabla.index = pd.MultiIndex.from_arrays(
            [
                [
                    f"{self.proyecto}/{_nombre_archivo(ruta)}"
                    for ruta in archivos
                ],
                tabla.index.get_level_values("start_time"),
                tabla.index.get_level_values("end_time"),
            ],
            names=["file", "start_time", "end_time"],
        )
        return tabla

    def audio_disponible(self, indice):
        return _nombre_archivo(indice[0]) in self.wav_por_nombre

    def _indice_audio_absoluto(self, tabla):
        tabla_audio = tabla.copy()
        tabla_audio.index = pd.MultiIndex.from_arrays(
            [
                [
                    str((self.raiz / Path(ruta)).resolve())
                    for ruta in tabla.index.get_level_values("file")
                ],
                tabla.index.get_level_values("start_time"),
                tabla.index.get_level_values("end_time"),
            ],
            names=["file", "start_time", "end_time"],
        )
        return tabla_audio

    def cargar_etiquetas(self, regenerar=False):
        if self.ruta_etiquetas.exists() and not regenerar:
            etiquetas = pd.read_csv(self.ruta_etiquetas, index_col=[0, 1, 2])
            etiquetas.index.names = ["file", "start_time", "end_time"]
            etiquetas = self._normalizar_indice(etiquetas)
            print(f"Se reutilizó {self.ruta_etiquetas}")
        else:
            etiquetas = self._generar_etiquetas()
            etiquetas.to_csv(self.ruta_etiquetas)
            print(f"Etiquetas guardadas en {self.ruta_etiquetas}")

        if etiquetas.index.names != ["file", "start_time", "end_time"]:
            raise ValueError("El índice de etiquetas no tiene el formato esperado.")
        if not etiquetas.index.is_unique:
            raise ValueError("El índice de etiquetas contiene ventanas duplicadas.")
        if not set(np.unique(etiquetas.to_numpy())).issubset({0, 1}):
            raise ValueError("Las etiquetas deben ser binarias.")
        return etiquetas

    def _generar_etiquetas(self):
        filas = []
        for sonido in self.sonidos_proyecto.itertuples(index=False):
            numero_segmentos = int(float(sonido.duration) // self.duracion_segmento)
            for indice_segmento in range(numero_segmentos):
                inicio_segmento = indice_segmento * self.duracion_segmento
                for desplazamiento in (0.0, 5.0):
                    filas.append(
                        {
                            "sound_id": sonido.id,
                            "audio_file": sonido.audio_file,
                            "start_time": inicio_segmento + desplazamiento,
                            "end_time": inicio_segmento + desplazamiento + 5.0,
                        }
                    )

        ventanas = pd.DataFrame(filas).sort_values(["sound_id", "start_time"])
        ventanas["file"] = ventanas["audio_file"].map(
            lambda nombre: f"{self.proyecto}/{nombre}"
        )
        anotaciones = self.anotaciones.loc[
            self.anotaciones["sound_id"].isin(set(ventanas["sound_id"]))
        ].copy()
        clases = sorted(anotaciones["category"].unique())
        matriz = np.zeros((len(ventanas), len(clases)), dtype="uint8")
        columna_clase = {clase: posicion for posicion, clase in enumerate(clases)}
        posiciones_por_sonido = ventanas.groupby("sound_id", sort=False).indices

        for evento in anotaciones.itertuples(index=False):
            posiciones = np.asarray(posiciones_por_sonido[evento.sound_id])
            candidatas = ventanas.iloc[posiciones]
            solapa = (
                candidatas["start_time"].to_numpy() < float(evento.t_max)
            ) & (candidatas["end_time"].to_numpy() > float(evento.t_min))
            matriz[posiciones[solapa], columna_clase[evento.category]] = 1

        indice = pd.MultiIndex.from_frame(
            ventanas[["file", "start_time", "end_time"]],
            names=["file", "start_time", "end_time"],
        )
        return pd.DataFrame(matriz, index=indice, columns=clases)

    def cargar_embeddings(self, etiquetas, regenerar=False):
        if self.ruta_embeddings.exists() and not regenerar:
            embeddings = pd.read_csv(
                self.ruta_embeddings, index_col=[0, 1, 2]
            )
            embeddings.index.names = ["file", "start_time", "end_time"]
            embeddings = self._normalizar_indice(embeddings)
            print(f"Se reutilizó {self.ruta_embeddings}")
            return embeddings

        import bioacoustics_model_zoo as bmz

        requeridos = {
            _nombre_archivo(ruta)
            for ruta in etiquetas.index.get_level_values("file").unique()
        }
        faltantes = sorted(requeridos.difference(self.wav_por_nombre))
        if faltantes:
            raise FileNotFoundError(
                "Para generar nuevamente los embeddings se necesitan todos los WAV. "
                "Usa descargar_audio=True o conserva el CSV de embeddings existente. "
                f"Faltan {len(faltantes)} archivos; primer ejemplo: {faltantes[0]}"
            )

        etiquetas_audio = self._indice_audio_absoluto(etiquetas[[]])
        inaccesibles = [
            ruta
            for ruta in etiquetas_audio.index.get_level_values("file").unique()
            if not Path(ruta).exists()
        ]
        if inaccesibles:
            raise FileNotFoundError(
                "No se pueden generar embeddings. "
                f"Primer WAV inaccesible: {inaccesibles[0]}"
            )

        embeddings = bmz.Perch2().embed(
            etiquetas_audio,
            batch_size=32,
            num_workers=0,
        )
        embeddings = self._normalizar_indice(embeddings)
        embeddings.to_csv(self.ruta_embeddings)
        print(f"Embeddings guardados en {self.ruta_embeddings}")
        return embeddings

    def cargar_ventana(self, indice):
        ruta_relativa, inicio, fin = indice
        nombre = _nombre_archivo(ruta_relativa)
        if nombre not in self.wav_por_nombre:
            raise FileNotFoundError(
                f"El WAV {nombre} no está disponible para esta visualización."
            )
        frecuencia_muestreo, audio = wavfile.read(
            self.wav_por_nombre[nombre], mmap=True
        )
        muestra_inicial = round(float(inicio) * frecuencia_muestreo)
        muestra_final = round(float(fin) * frecuencia_muestreo)
        fragmento = np.asarray(audio[muestra_inicial:muestra_final])
        if fragmento.ndim > 1:
            fragmento = fragmento.astype("float32").mean(axis=1)
        if np.issubdtype(fragmento.dtype, np.integer):
            fragmento = fragmento.astype("float32") / max(
                abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max
            )
        else:
            fragmento = fragmento.astype("float32")
        return frecuencia_muestreo, fragmento

    def eventos_ventana(self, indice, categorias):
        ruta_relativa, inicio, fin = indice
        sound_id = self.sonido_id_por_archivo[Path(ruta_relativa).name]
        return self.anotaciones.loc[
            self.anotaciones["sound_id"].eq(sound_id)
            & self.anotaciones["category"].isin(categorias)
            & self.anotaciones["t_min"].lt(float(fin))
            & self.anotaciones["t_max"].gt(float(inicio))
        ]


def alinear_datos(etiquetas, embeddings):
    indice_comun = etiquetas.index.intersection(embeddings.index, sort=False)
    if len(indice_comun) == 0:
        raise ValueError("Etiquetas y embeddings no comparten ninguna ventana.")

    descartadas = (
        len(etiquetas) - len(indice_comun),
        len(embeddings) - len(indice_comun),
    )
    if descartadas != (0, 0):
        warnings.warn(
            f"Se descartarán {descartadas[0]} etiquetas y "
            f"{descartadas[1]} embeddings sin pareja."
        )

    etiquetas = etiquetas.loc[indice_comun].sort_index()
    embeddings = embeddings.loc[indice_comun].sort_index()
    if not etiquetas.index.equals(embeddings.index):
        raise ValueError("No fue posible alinear etiquetas y embeddings.")
    if not np.isfinite(embeddings.to_numpy()).all():
        raise ValueError("Los embeddings contienen valores no finitos.")
    return etiquetas, embeddings


@dataclass
class Particion:
    etiquetas_entrenamiento: pd.DataFrame
    etiquetas_prueba: pd.DataFrame
    embeddings_entrenamiento: pd.DataFrame
    embeddings_prueba: pd.DataFrame
    grabaciones: pd.Series

    def datos_binarios(self, especie):
        y_entrenamiento = self.etiquetas_entrenamiento[especie].astype("uint8")
        y_prueba = self.etiquetas_prueba[especie].astype("uint8")
        if y_entrenamiento.nunique() < 2 or y_prueba.nunique() < 2:
            raise ValueError(
                "La especie debe tener casos positivos y negativos en ambos conjuntos."
            )
        return (
            self.embeddings_entrenamiento.to_numpy(dtype="float32"),
            self.embeddings_prueba.to_numpy(dtype="float32"),
            y_entrenamiento,
            y_prueba,
        )


def separar_por_grabacion(etiquetas, embeddings, especie, semilla=42):
    if especie not in etiquetas.columns:
        raise KeyError(f"La especie {especie} no aparece en las etiquetas.")

    grabaciones = pd.Series(
        etiquetas.index.get_level_values("file").map(lambda ruta: Path(ruta).name),
        index=etiquetas.index,
        name="grabacion",
    )
    particion = None
    for intento in range(100):
        separador = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=semilla + intento,
        )
        entrenamiento, prueba = next(
            separador.split(etiquetas, etiquetas[especie], groups=grabaciones)
        )
        if (
            etiquetas.iloc[entrenamiento][especie].nunique() == 2
            and etiquetas.iloc[prueba][especie].nunique() == 2
        ):
            particion = (entrenamiento, prueba)
            break

    if particion is None:
        raise ValueError(
            f"No se encontró una separación válida para {especie}; "
            "elige otra especie."
        )

    entrenamiento, prueba = particion
    grabaciones_entrenamiento = set(grabaciones.iloc[entrenamiento])
    grabaciones_prueba = set(grabaciones.iloc[prueba])
    if not grabaciones_entrenamiento.isdisjoint(grabaciones_prueba):
        raise ValueError("Una grabación aparece en entrenamiento y prueba.")

    return Particion(
        etiquetas.iloc[entrenamiento],
        etiquetas.iloc[prueba],
        embeddings.iloc[entrenamiento],
        embeddings.iloc[prueba],
        grabaciones,
    )


def resumen_clases(y_entrenamiento, y_prueba):
    resumen = pd.DataFrame(
        {
            "entrenamiento": y_entrenamiento.value_counts(),
            "prueba": y_prueba.value_counts(),
        }
    ).fillna(0).astype(int)
    resumen.index = resumen.index.map({0: "ausente", 1: "presente"})
    return resumen


def crear_modelos(semilla=42):
    return {
        "Regresión logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=semilla,
            ),
        ),
        "SVM lineal": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                C=0.025,
                class_weight="balanced",
                random_state=semilla,
            ),
        ),
        "SVM RBF": make_pipeline(
            StandardScaler(),
            SVC(
                gamma="scale",
                C=1,
                class_weight="balanced",
                random_state=semilla,
            ),
        ),
        "3 vecinos más cercanos": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
        "Bosque aleatorio": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=semilla,
        ),
        "Naive Bayes gaussiano": make_pipeline(StandardScaler(), GaussianNB()),
        "SGD logístico": make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=2000,
                tol=1e-3,
                class_weight="balanced",
                random_state=semilla,
            ),
        ),
    }


def _puntaje_continuo(modelo, X):
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X)
        posicion = np.flatnonzero(np.asarray(modelo.classes_) == 1)
        if len(posicion) != 1:
            raise ValueError("El modelo no contiene una única clase positiva 1.")
        return probabilidades[:, posicion[0]]
    if hasattr(modelo, "decision_function"):
        return np.asarray(modelo.decision_function(X)).ravel()
    raise TypeError(
        f"{type(modelo).__name__} no ofrece predict_proba ni decision_function."
    )


@dataclass
class ComparacionModelos:
    tabla: pd.DataFrame
    modelos: dict
    puntajes_prueba: dict

    @property
    def mejor_modelo(self):
        return self.tabla.index[0]

    @property
    def mejor_puntaje(self):
        return self.puntajes_prueba[self.mejor_modelo]


def comparar_modelos(
    modelos,
    X_entrenamiento,
    y_entrenamiento,
    X_prueba,
    y_prueba,
):
    resultados = []
    modelos_ajustados = {}
    puntajes_prueba = {}

    for nombre, modelo_base in modelos.items():
        modelo = clone(modelo_base)
        inicio = perf_counter()
        modelo.fit(X_entrenamiento, y_entrenamiento)
        segundos = perf_counter() - inicio
        puntaje_entrenamiento = _puntaje_continuo(modelo, X_entrenamiento)
        puntaje_prueba = _puntaje_continuo(modelo, X_prueba)
        resultados.append(
            {
                "modelo": nombre,
                "AP entrenamiento": average_precision_score(
                    y_entrenamiento, puntaje_entrenamiento
                ),
                "AP prueba": average_precision_score(y_prueba, puntaje_prueba),
                "ROC AUC entrenamiento": roc_auc_score(
                    y_entrenamiento, puntaje_entrenamiento
                ),
                "ROC AUC prueba": roc_auc_score(y_prueba, puntaje_prueba),
                "segundos de ajuste": segundos,
            }
        )
        modelos_ajustados[nombre] = modelo
        puntajes_prueba[nombre] = puntaje_prueba

    tabla = (
        pd.DataFrame(resultados)
        .set_index("modelo")
        .sort_values("AP prueba", ascending=False)
    )
    return ComparacionModelos(tabla, modelos_ajustados, puntajes_prueba)


def graficar_curvas(y_prueba, puntaje, nombre_modelo, especie):
    precision, exhaustividad, _ = precision_recall_curve(y_prueba, puntaje)
    falsos_positivos, verdaderos_positivos, _ = roc_curve(y_prueba, puntaje)
    prevalencia = y_prueba.mean()

    _, ejes = plt.subplots(1, 2, figsize=(13, 5))
    ejes[0].plot(exhaustividad, precision, label=nombre_modelo)
    ejes[0].axhline(
        prevalencia,
        color="gray",
        linestyle="--",
        label=f"Prevalencia = {prevalencia:.3f}",
    )
    ejes[0].set(
        xlabel="Exhaustividad",
        ylabel="Precisión",
        title=f"Curva precisión–exhaustividad: {especie}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ejes[0].legend()

    ejes[1].plot(falsos_positivos, verdaderos_positivos, label=nombre_modelo)
    ejes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Azar")
    ejes[1].set(
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
        title=f"Curva ROC: {especie}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ejes[1].legend()
    plt.tight_layout()


def seleccionar_umbral(y_prueba, puntaje):
    precision, exhaustividad, umbrales = precision_recall_curve(y_prueba, puntaje)
    f1 = 2 * precision[:-1] * exhaustividad[:-1] / (
        precision[:-1] + exhaustividad[:-1] + 1e-12
    )
    posicion = int(np.nanargmax(f1))
    umbral = umbrales[posicion]
    prediccion = (puntaje >= umbral).astype("uint8")
    reporte = classification_report(
        y_prueba,
        prediccion,
        target_names=["ausente", "presente"],
        digits=3,
        zero_division=0,
    )
    return umbral, prediccion, reporte


def resumir_etiquetas(etiquetas, proyecto, taxonomia):
    frecuencias = etiquetas.sum().sort_values(ascending=False)
    nombres_cientificos = taxonomia.set_index("code")["species"].to_dict()
    frecuencias_mostradas = frecuencias.rename(
        index=lambda codigo: nombres_cientificos.get(codigo, codigo)
    )
    frecuencias_mostradas.index.name = "especie"
    display(
        frecuencias_mostradas.rename("ventanas_positivas").to_frame().head(15)
    )
    _, eje = plt.subplots(figsize=(10, 6))
    frecuencias_mostradas.head(15).sort_values().plot.barh(
        ax=eje, color="#2563eb"
    )
    eje.set(
        xlabel="Ventanas positivas de 5 s",
        ylabel="Nombre científico",
        title=f"Las 15 especies más frecuentes en {proyecto}",
    )
    eje.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()

    etiquetas_por_ventana = etiquetas.sum(axis=1)
    print("Ventanas sin etiqueta:", int((etiquetas_por_ventana == 0).sum()))
    print(
        "Ventanas con más de una etiqueta:",
        int((etiquetas_por_ventana > 1).sum()),
    )
    return frecuencias


def _dibujar_ventana(
    dataset,
    indice,
    presentes,
    color,
    titulo,
    titulo_espectrograma,
):
    ruta_relativa, inicio, fin = indice
    frecuencia_muestreo, fragmento = dataset.cargar_ventana(indice)
    tiempo = np.arange(len(fragmento)) / frecuencia_muestreo
    eventos = dataset.eventos_ventana(indice, presentes)
    limite_frecuencia = min(frecuencia_muestreo / 2, 24_000)

    display(Audio(fragmento, rate=frecuencia_muestreo, normalize=False))
    figura, ejes = plt.subplots(
        2, 1, figsize=(12, 7), constrained_layout=True
    )
    ejes[0].plot(tiempo, fragmento, color=color, linewidth=0.7)
    ejes[0].set(
        xlim=(0, float(fin) - float(inicio)),
        xlabel="Tiempo dentro de la ventana (s)",
        ylabel="Amplitud",
        title=titulo,
    )
    espectro, frecuencias, _, imagen = ejes[1].specgram(
        fragmento,
        NFFT=4096,
        Fs=frecuencia_muestreo,
        noverlap=3072,
        cmap="inferno",
        scale="dB",
    )
    espectro_db = 10 * np.log10(
        np.maximum(espectro, np.finfo(float).tiny)
    )
    espectro_visible = espectro_db[
        frecuencias <= limite_frecuencia
    ]
    valores_finitos = espectro_visible[np.isfinite(espectro_visible)]
    if valores_finitos.size:
        minimo_percentil, maximo = np.percentile(
            valores_finitos, [20, 99.5]
        )
        minimo = max(minimo_percentil, maximo - 50)
        if maximo > minimo:
            imagen.set_clim(minimo, maximo)
    ejes[1].set(
        xlim=(0, float(fin) - float(inicio)),
        ylim=(0, limite_frecuencia),
        xlabel="Tiempo dentro de la ventana (s)",
        ylabel="Frecuencia (Hz)",
        title=titulo_espectrograma,
    )

    mapa_colores = plt.get_cmap("tab10")
    colores = {
        categoria: mapa_colores(posicion % 10)
        for posicion, categoria in enumerate(presentes)
    }
    dibujadas = set()
    for evento in eventos.itertuples(index=False):
        x0 = max(float(evento.t_min), float(inicio)) - float(inicio)
        x1 = min(float(evento.t_max), float(fin)) - float(inicio)
        y0 = max(float(evento.f_min), 0)
        y1 = min(float(evento.f_max), limite_frecuencia)
        if x1 > x0 and y1 > y0:
            ejes[1].add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=colores[evento.category],
                    linewidth=2,
                    label=(
                        evento.category
                        if evento.category not in dibujadas
                        else "_nolegend_"
                    ),
                )
            )
            dibujadas.add(evento.category)
    if dibujadas:
        ejes[1].legend(title="Especies", loc="upper right", framealpha=0.9)
    display(figura)
    plt.close(figura)


def crear_explorador_etiquetas(dataset, etiquetas, frecuencias):
    nombres = dataset.taxonomia.set_index("code")["species"].to_dict()

    def indices_positivos(especie):
        return [
            indice
            for indice in etiquetas.index[etiquetas[especie].eq(1)]
            if dataset.audio_disponible(indice)
        ]

    especies_disponibles = [
        codigo for codigo in frecuencias.index if indices_positivos(codigo)
    ]
    if not especies_disponibles:
        return widgets.HTML(
            "No hay WAV disponibles para el explorador de etiquetas."
        )

    selector_especie = widgets.Dropdown(
        options=[
            (
                f"{codigo} — {nombres.get(codigo, 'sin nombre')} "
                f"({int(frecuencias[codigo])} ventanas)",
                codigo,
            )
            for codigo in especies_disponibles
        ],
        value=especies_disponibles[0],
        description="Especie:",
        layout=widgets.Layout(width="650px"),
        style={"description_width": "initial"},
    )
    selector_ejemplo = widgets.BoundedIntText(
        value=1,
        min=1,
        max=len(indices_positivos(selector_especie.value)),
        description="Ejemplo:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "initial"},
    )
    salida = widgets.Output()

    def mostrar(*_):
        especie = selector_especie.value
        ejemplos = indices_positivos(especie)
        posicion = min(selector_ejemplo.value, len(ejemplos)) - 1
        indice = ejemplos[posicion]
        ruta_relativa, inicio, fin = indice
        presentes = etiquetas.loc[indice]
        presentes = presentes.index[presentes.eq(1)].tolist()
        salida.clear_output(wait=True)
        with salida:
            print(
                f"{Path(ruta_relativa).name} | {inicio:.1f}–{fin:.1f} s | "
                f"ejemplo {posicion + 1} de {len(ejemplos)}"
            )
            print("Etiquetas presentes:", ", ".join(presentes))
            _dibujar_ventana(
                dataset,
                indice,
                presentes,
                "#2563eb",
                "Forma de onda",
                "Espectrograma y todos los eventos etiquetados",
            )

    def cambiar_especie(cambio):
        if cambio["name"] == "value":
            selector_ejemplo.unobserve(mostrar, names="value")
            try:
                selector_ejemplo.max = len(indices_positivos(cambio["new"]))
                selector_ejemplo.value = 1
            finally:
                selector_ejemplo.observe(mostrar, names="value")
            mostrar()

    selector_especie.observe(cambiar_especie, names="value")
    selector_ejemplo.observe(mostrar, names="value")
    mostrar()
    return widgets.VBox([selector_especie, selector_ejemplo, salida])


def crear_explorador_errores(
    dataset,
    etiquetas_prueba,
    y_prueba,
    prediccion,
    puntaje,
    especie,
    umbral,
):
    tabla = pd.DataFrame(
        {
            "real": y_prueba.to_numpy(),
            "prediccion": prediccion,
            "puntaje": puntaje,
        },
        index=y_prueba.index,
    )
    errores = {
        "Falso positivo": tabla.loc[
            tabla["real"].eq(0) & tabla["prediccion"].eq(1)
        ].sort_values("puntaje", ascending=False),
        "Falso negativo": tabla.loc[
            tabla["real"].eq(1) & tabla["prediccion"].eq(0)
        ].sort_values("puntaje"),
    }
    resumen = pd.Series(
        {tipo: len(filas) for tipo, filas in errores.items()},
        name="ventanas",
    )
    errores_disponibles = {
        tipo: filas.loc[
            [dataset.audio_disponible(indice) for indice in filas.index]
        ]
        for tipo, filas in errores.items()
    }
    opciones = [
        (f"{tipo} ({len(filas)})", tipo)
        for tipo, filas in errores_disponibles.items()
        if len(filas) > 0
    ]
    if not opciones:
        return resumen, widgets.HTML(
            "No hay WAV conservados para visualizar los errores del modelo."
        )

    selector_tipo = widgets.Dropdown(
        options=opciones,
        description="Tipo de error:",
        layout=widgets.Layout(width="420px"),
        style={"description_width": "initial"},
    )
    selector_ejemplo = widgets.BoundedIntText(
        value=1,
        min=1,
        max=len(errores_disponibles[selector_tipo.value]),
        description="Ejemplo:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "initial"},
    )
    salida = widgets.Output()

    def mostrar(*_):
        tipo = selector_tipo.value
        casos = errores_disponibles[tipo]
        posicion = min(selector_ejemplo.value, len(casos)) - 1
        indice = casos.index[posicion]
        caso = casos.iloc[posicion]
        ruta_relativa, inicio, fin = indice
        presentes = etiquetas_prueba.loc[indice]
        presentes = presentes.index[presentes.eq(1)].tolist()
        salida.clear_output(wait=True)
        with salida:
            print(
                f"{tipo} {posicion + 1} de {len(casos)} | "
                f"{Path(ruta_relativa).name} | {inicio:.1f}–{fin:.1f} s"
            )
            print(
                f"Especie objetivo: {especie} | real={int(caso['real'])} | "
                f"predicción={int(caso['prediccion'])} | "
                f"puntaje={caso['puntaje']:.4f} | umbral={umbral:.4f}"
            )
            print(
                "Etiquetas presentes:",
                ", ".join(presentes) if presentes else "ninguna",
            )
            _dibujar_ventana(
                dataset,
                indice,
                presentes,
                "#dc2626",
                f"Forma de onda — {tipo}",
                "Espectrograma y eventos realmente anotados",
            )

    def cambiar_tipo(cambio):
        if cambio["name"] == "value":
            selector_ejemplo.unobserve(mostrar, names="value")
            try:
                selector_ejemplo.max = len(errores_disponibles[cambio["new"]])
                selector_ejemplo.value = 1
            finally:
                selector_ejemplo.observe(mostrar, names="value")
            mostrar()

    selector_tipo.observe(cambiar_tipo, names="value")
    selector_ejemplo.observe(mostrar, names="value")
    mostrar()
    controles = widgets.HBox([selector_tipo, selector_ejemplo])
    return resumen, widgets.VBox([controles, salida])
