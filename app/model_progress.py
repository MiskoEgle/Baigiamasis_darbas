import threading
import time

class ModelProgress:
    """
    Klasė, skirta sekti modelių mokymo progresą.
    Leidžia stebėti ir valdyti skirtingų modelių (KNN, CNN, Transformer) mokymo būseną.
    """
    def __init__(self):
        # Sukuriame užraktą (lock) apsaugai nuo vienu metu vykstančių operacijų
        self.lock = threading.Lock()
        # Inicializuojame progreso žodyną kiekvienam modeliui
        self.progress = {
            'knn': self._empty_progress(),
            'cnn': self._empty_progress(),
            'transformer': self._empty_progress()
        }
        # Inicializuojame sustabdymo vėliavėles kiekvienam modeliui
        self.stop_flags = {
            'knn': False,
            'cnn': False,
            'transformer': False
        }
    def _empty_progress(self):
        """
        Sukuria tuščią progreso žodyną su numatytomis reikšmėmis.
        """
        return {
            'current': 0,        # Dabartinis progresas
            'total': 0,          # Bendras darbų kiekis
            'percent': 0,        # Progreso procentai
            'start_time': None,  # Pradžios laikas
            'eta': None,         # Numatomas likęs laikas (Estimated Time of Arrival)
            'status': 'idle',    # Dabartinė būsena
            'message': ''        # Žinutė apie dabartinį veikimą
        }
    def start(self, model_type, total):
        """
        Pradeda progreso sekimą nurodytam modeliui.
        
        Args:
            model_type: Modelio tipas ('knn', 'cnn' arba 'transformer')
            total: Bendras darbų kiekis
        """
        with self.lock:
            self.progress[model_type] = {
                'current': 0,
                'total': total,
                'percent': 0,
                'start_time': time.time(),
                'eta': None,
                'status': 'in_progress',
                'message': ''
            }
            self.stop_flags[model_type] = False
    def update(self, model_type, current, message=''):
        """
        Atnaujina modelio progreso informaciją.
        
        Args:
            model_type: Modelio tipas
            current: Dabartinis progresas
            message: Žinutė apie dabartinį veikimą
        """
        with self.lock:
            prog = self.progress[model_type]
            prog['current'] = current
            prog['percent'] = int(100 * current / prog['total']) if prog['total'] else 0
            # Skaičiuojame praėjusį laiką
            elapsed = time.time() - prog['start_time'] if prog['start_time'] else 0
            # Skaičiuojame numatomą likusį laiką
            if current > 0 and elapsed > 0:
                eta = elapsed * (prog['total'] - current) / current
                prog['eta'] = int(eta)
            else:
                prog['eta'] = None
            prog['message'] = message
    def finish(self, model_type):
        """
        Pažymi modelio mokymą kaip užbaigtą.
        
        Args:
            model_type: Modelio tipas
        """
        with self.lock:
            prog = self.progress[model_type]
            prog['current'] = prog['total']
            prog['percent'] = 100
            prog['eta'] = 0
            prog['status'] = 'completed'
            self.stop_flags[model_type] = False
    def error(self, model_type, message):
        """
        Pažymi modelio mokymą kaip nepavykusį.
        
        Args:
            model_type: Modelio tipas
            message: Klaidos žinutė
        """
        with self.lock:
            prog = self.progress[model_type]
            prog['status'] = 'error'
            prog['message'] = message
            self.stop_flags[model_type] = False
    def stop(self, model_type):
        """
        Pradeda modelio mokymo sustabdymo procesą.
        
        Args:
            model_type: Modelio tipas
        """
        with self.lock:
            self.stop_flags[model_type] = True
            prog = self.progress[model_type]
            prog['status'] = 'stopping'
            prog['message'] = 'Stopping training...'
    def should_stop(self, model_type):
        """
        Patikrina ar modelis turėtų būti sustabdytas.
        
        Args:
            model_type: Modelio tipas
        Returns:
            bool: True jei modelis turėtų būti sustabdytas
        """
        with self.lock:
            return self.stop_flags[model_type]
    def get(self, model_type):
        """
        Grąžina nurodyto modelio progreso informaciją.
        
        Args:
            model_type: Modelio tipas
        Returns:
            dict: Modelio progreso informacija
        """
        with self.lock:
            return dict(self.progress[model_type])
    def get_all(self):
        """
        Grąžina visų modelių progreso informaciją.
        
        Returns:
            dict: Visų modelių progreso informacija
        """
        with self.lock:
            return {k: dict(v) for k, v in self.progress.items()}

model_progress = ModelProgress() 