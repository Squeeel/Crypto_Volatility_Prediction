import numpy as np
import pandas as pd
from typing import Union, Any, Optional, Tuple
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import warnings

# Filtre global ciblé pour tuer le warning de layout de Matplotlib
# (fonctionne même lors du rendu différé par IPython/Jupyter)
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*This figure includes Axes that are not compatible with tight_layout.*"
)

########################################################################################################################
# compute_backtest retourne un objet de type Backtest
########################################################################################################################
'''compute_backtest(...) -> Backtest:

###############
# input:
###############

> pos_df: Union[pd.DataFrame, pd.core.series.Series]:
    Dataframe (ou series pandas si 1D) des positions avec actifs sur les colonnes.
    ATTENTION : Si compound_returns=True, ces valeurs doivent représenter des poids d'exposition (ex: 1.0 pour 100%).

> perf_df: Union[pd.DataFrame, pd.core.series.Series]:
    Dataframe (ou series pandas si 1D) des rendements. Actifs sur les colonnes.
    ATTENTION à la nature des rendements :
    - Si perf_transform=True, la série (exprimant des prix) sera transformée en rendements relatifs via pct_change().
    - Si compound_returns=True, ces rendements DOIVENT impérativement être des rendements relatifs en décimales (ex: 0.05 = 5%).
    - Si compound_returns=False, vous pouvez passer des différences de prix brutes (unités monétaires, ex: +5$) 
      SI ET SEULEMENT SI les positions représentent un nombre nominal d'actions ou de contrats.

> transaction_costs_bps: float = 0.0:
    Coûts de transaction exprimés en points de base (bps) prélevés à chaque trade (indexés sur le turnover). 
    Si > 0, ces frais seront déduits de la courbe du PnL.

> compound_returns: bool = False:
    - Si False, calcule un PnL additif classique (dot product : somme des returns * positions).
    - Si True, calcule la courbe d'équité en composition (cumprod(1 + net_return)), simulant l'évolution réelle d'un capital réinvesti.

> risk_free_rate_pct: float = 0.0:
    Taux sans risque annualisé exprimé en pourcentage (ex: 2.5 pour 2.5%). 
    S'il est différent de 0, il sera soustrait du rendement annualisé lors du calcul du ratio de Sharpe pour évaluer 
    la véritable surperformance de la stratégie.

> position_lag_value: int = 1:
    Valeur entière du lag.

> position_lag_unit: str = 'units':
    Unité du lag : si 'units', utilise le time step de la série d'entrée, sinon n'importe quelle unité de temps
    compréhensible par pandas (cf : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html).
    Ex : "min", "hours", "days", etc.

    ATTENTION : Dans cette version, le Backtester ne gère les calculs que jusqu'à la fréquence/période minimale de 1 sec.

> tolerance_method: str = 'previous':
    Méthode de recherche utilisée pour récupérer la position à mettre en face de la performance (si les timestamps des deux,
    avec lag pour les positions, ne sont pas alignés). Valeurs possibles :
    - "previous": on recherche la position avant le timestamp de la performance.
    - "exact": on n'autorise qu'un matching exact entre le timestamp de la position (éventuellement laguée) et celui de la perf.
    - "nearest": on recherche la position la plus proche possible dans le temps, elle peut être avant ou après le timestamp de la perf.
    - "next": on recherche la position après le timestamp de la perf.

    ATTENTION : seules les méthodes 'previous' et 'exact' permettent de s'assurer qu'il n'y a pas de forward trading.

> tolerance_value: Optional[float] = None:
    - Si tolerance_value != None, on va chercher une position éloignée au plus de tolerance_value tolerance_unit
      du timestamp de la performance. Si aucune position proche n'est trouvée, on renvoie np.nan.
    - Si tolerance_value == None, on ira chercher aussi loin que nécessaire (ex : en arrière si tolerance_method = 'previous').

> tolerance_unit: str = 'minutes':
    Unité de temps pour tolerance_value.

> position_fill_value: Optional[float] = None:
    Valeur pour remplir les positions manquantes (np.nan) :
    - Si position_fill_value == None, on applique un forward fill (ffill).
    - Si position_fill_value != None, on remplit avec cette valeur.

> perf_transform: bool = False:
    - Si True, transforme perf_df (représente des prix) en performances : perf(t) = prix(t)/prix(t-1) - 1.
    - Si False, perf_df représente déjà des performances.

> perf_fill_value: Optional[float] = 0:
    Valeur pour remplir les performances manquantes (np.nan).

> detail_LS: bool = False:
    - Si True, calcule le split entre le PnL "Long" (positions longues) et "Short" (positions courtes).
    - Si False, pas de split.

> detail_assets: bool = False:
    - Si True, calcule le détail du PnL par actif.
    - Si False, calcule uniquement le PnL total.

> show_total: bool = True:
    - Si True, affiche le PnL total.
    - Si False, n'affiche pas le PnL (utile si detail_LS ou detail_assets est False).

> compute_max_DD: bool = True:
    - Si True, calcule le maximum drawdown (DD).
    - Si False, pas de calcul du max DD.

> show_pnl: bool = True:
    - Si True, affiche le graphe du PnL cumulé.
    - Si False, n'affiche pas le graphe.

> compute_DD_points: bool = False:
    - Si True, affiche les points de départ et d'arrivée du max DD.
    - Si False, ne les affiche pas.

> title: str = '':
    Titre du Backtest.

> check_index: bool = False:
    - Si True, vérifie que pos_df et perf_df ont le même index pandas, sinon réindexe pos_df sur perf_df.
    - Si False, aucune vérification des index.

> sort_by: Optional[str] = 'sharpe':
    Métrique utilisée pour trier les actifs (si detail_assets est True). Métriques possibles :
    "sharpe", "PnL/Trade", "holding period", "return (yearly)", "$\\sigma$ PnL", "max DD",
    "relative max DD (in $\\sigma$s)", "%time position".

> sort_ascending: bool = False:
    Indique si le tri des actifs par la métrique sort_by est ascendant (par défaut, False = tri descendant).

> display_nb_timestamps: int = 2560:
    Nombre maximum de points (timestamps) conservés pour les séries PnL et positions dans l'objet retourné 
    (sous-échantillonnage pour un affichage plus rapide). Si <= 0, aucun sous-échantillonnage n'est effectué.

> improve_subsampled_position_display : bool = False, optional
    Si True et qu'un sous-échantillonnage est actif, recherche les trades masqués par le sous-échantillonnage 
    et force leur inclusion pour garantir la visibilité des pics de position. Default: False.

> improve_subsampled_position_additional_timestamps_multiplier : float = 2.0, optional
    Multiplicateur définissant le "budget" de points additionnels alloué à la récupération des trades perdus. 
    (ex: 2.0 = on autorise jusqu'à 2 * display_nb_timestamps points en plus). Default: 2.0.

###############
# output:
###############
> Backtest: objet contenant les métriques de performance et les données de performance/positions à afficher.
'''

########################################################################################################################
# plot_backtest affiche les métriques et le graphe des performances/positions d'un objet de type Backtest
########################################################################################################################
'''plot_backtest(...) -> None:

###############
# input:
###############

> bt: Backtest:
    Un objet Backtest issu de compute_Backtest(...).

> fig_width: float = 20:
    Largeur de la figure en unités matplotlib (inches).

> fig_height: float = 15:
    Hauteur de la figure en unités matplotlib (inches).

> font_size: float = 10:
    Taille de la police utilisée pour les titres/tableaux.

> table_row_height: float = 0.4:
    Hauteur d'une ligne du tableau des métriques de performance.

> title_vertical_space: float = 0.0:
    Espacement entre le titre et le tableau des métriques.

> dpi: Optional[int] = 72:
    Résolution (DPI) de l'affichage.

> show_fig: bool = True:
    - Si True, affiche le graphe avec matplotlib.
    - Si False, n'affiche pas le graphe.

> save_fig: bool = False:
    - Si True, enregistre l'image du Backtest.
    - Si False, pas d'enregistrement.

> save_path: Optional[str] = None:
    - Si None, enregistre dans le répertoire courant.
    - Sinon, enregistre dans le répertoire indiqué par save_path.

> default_save_extension: str = '.png':
    Extension du fichier pour l'image enregistrée.

> display_nb_timestamps: int = -1:
    Nombre maximum de points à afficher sur le graphe. Si le Backtest contient plus 
    de points, les séries sont sous-échantillonnées à la volée (seulement pour l'affichage).
    Si <= 0, affiche tous les points.

###############
# output:
###############
> Aucune sortie : l'affichage est réalisé graphiquement avec matplotlib.
'''

########################################################################################################################

def rd_unit(x: float, digits: int = 2) -> str:
    """
    Arrondit un nombre à un certain nombre de décimales et le convertit en chaîne de caractères.
    """
    return str(np.round(x, digits))

def rd_pct(x: float, digits: int = 2, string: str = '%') -> str:
    """
    Convertit un nombre en pourcentage arrondi avec un symbole.
    """
    return (str(np.round(x * 10 ** digits, digits)) + ' ' + string)

def rd_bps(x: float, digits: int = 2, string: str = 'bps') -> str:
    """
    Convertit un nombre en points de base (bps) arrondis.
    """
    return str(np.round(x * 10 ** 4, digits)) + ' ' + string

class DataMatrix:
    """
    Conteneur de données léger inspiré de pandas.DataFrame.

    Cette classe est conçue comme une alternative performante à pandas.DataFrame
    pour des opérations matricielles intensives, en évitant une partie de
    l'overhead de pandas. Elle n'inclut pas de contrôles de cohérence pour
    maximiser la vitesse.
    """

    def __init__(self, index: Optional[np.ndarray] = None, columns: Optional[np.ndarray] = None,
                 values: Optional[np.ndarray] = None):

        self.index = np.array([]) if index is None else index
        self.columns = np.array([]) if columns is None else columns
        self.values = np.array([[]]) if values is None else values
        self.shape = self.values.shape

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key: Union[tuple, slice, int, list]):
        if isinstance(key, tuple):
            return DataMatrix(self.index[key[0]], self.columns[key[1]], self.values[key])

        else:
            return DataMatrix(self.index[key], self.columns, self.values[key])

    def __setitem__(self, key: Union[tuple, slice, int, list], value: Any):
        self.values[key] = value

    def __delitem__(self, key: Union[tuple, slice, int, list]):
        if isinstance(key, tuple):
            self.index = np.delete(self.index, key[0])
            self.columns = np.delete(self.columns, key[1])
            self.values = np.delete(np.delete(self.values, key[0], axis=0), key[1], axis=1)

        else:
            self.index = np.delete(self.index, key)
            self.values = np.delete(self.values, key)
        # --- Modif: On met à jour la shape suite à la délétion
        self.shape = self.values.shape

    def toDataFrame(self) -> pd.DataFrame:
        """Convertit l'instance DataMatrix en pandas.DataFrame."""
        return pd.DataFrame(self.values, index=self.index.T, columns=self.columns)

class Backtest:
    """
    Objet conteneur pour les résultats d'un backtest.
    """

    def __init__(self, nb_assets: int = 0, title: Optional[str] = None, metrics: Optional[DataMatrix] = None,
                 bt_dict: Optional[dict] = None):
        self.nb_assets = nb_assets
        self.title = title
        self.metrics = metrics
        self.bt_dict = bt_dict

def is_leap_year(year: int) -> bool:
    """Indique si l'année est bisextile"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def seconds_to_dhms(time_in_seconds: int, nb_seconds_per_day: float = 86400.) -> Tuple[int, int, int, int]:
    """Convertit une durée en secondes en jours, heures, minutes, secondes."""
    nb_days = int(time_in_seconds // nb_seconds_per_day)
    remaining_time = time_in_seconds % nb_seconds_per_day
    nb_hours = int(remaining_time // 3600)
    remaining_time %= 3600
    nb_minutes = int(remaining_time // 60)
    final_seconds = int(remaining_time % 60)
    return nb_days, nb_hours, nb_minutes, final_seconds

def get_duration_details(end_timestamp: np.datetime64, start_timestamp: np.datetime64) -> \
        (float, Tuple[int]):
    """Calcule la différence en années flottantes et en (y,m,d,h,m,s) entre deux timestamps."""
    if end_timestamp < start_timestamp:
        start_timestamp, end_timestamp = end_timestamp, start_timestamp

    rd = relativedelta(end_timestamp, start_timestamp)
    nb_years: int = rd.years

    td_after_years = end_timestamp - (start_timestamp + relativedelta(years=nb_years))
    year_of_remaining_delta = start_timestamp.year + nb_years
    seconds_in_that_year: int = 31622400 if is_leap_year(year_of_remaining_delta) else 31536000

    float_years = nb_years + td_after_years.total_seconds() / seconds_in_that_year
    duration_tuple = (rd.years, rd.months, rd.days, rd.hours, rd.minutes, int(round(rd.seconds)))
    return float_years, duration_tuple

def format_days_timedelta(days: int, hours: int, minutes: int, seconds: int, days_subst: str = '') -> str:
    """Formate une durée en jours/heures/minutes/secondes."""
    parts: list = []
    if days:
        parts.append(f"{days} {days_subst}{'days' if days > 1 else 'day'}")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}min")
    if seconds or parts:
        parts.append(f"{seconds}s")
    return ' '.join(parts) or "0s"

def format_years_timedelta(years: int, months: int, days: int, hours: int, minutes: int, seconds: int,
                           days_subst: str = '') -> str:
    """Formate une durée en années/mois/jours/heures/minutes/secondes."""
    parts: list = []
    if years:
        parts.append(f"{years} {'years' if years > 1 else 'year'}")
    if months:
        parts.append(f"{months} {'months' if months > 1 else 'month'}")

    dtd = format_days_timedelta(days, hours, minutes, seconds, days_subst)
    if (parts and dtd != "Os") or not parts:
        parts.append(dtd)

    return ' '.join(parts)

def format_timedelta_str(years:int=0, months:int=0, days:int=0, hours:int=0, minutes:int=0, seconds:int=0) -> str:
    """Formate une durée en une chaîne de caractères lisible."""
    parts = []
    if years: parts.append(f"{years} {'years' if years > 1 else 'year'}")
    if months: parts.append(f"{months} {'months' if months > 1 else 'month'}")
    if days: parts.append(f"{days} {'days' if days > 1 else 'day'}")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}min")
    if seconds or not parts: parts.append(f"{seconds}s")
    return " ".join(parts)

def build_backtest_title(custom_title: str, NbAssets: int, position_lag_value: int, position_lag_units: str,
                         tolerance_method: str, tolerance_value: Optional[int], tolerance_units: str,
                         title_duration: str, transaction_costs_bps: float, total_tc_yearly_pct: float,
                         compound_returns: bool, risk_free_rate_pct: float, 
                         total_sharpe_old: float, total_sharpe_new: float) -> str:
    """Constitue le titre complet du backtest (métriques, lags, coûts de transaction, rendements et RFR)."""
    
    # 1. Ajout du titre personnalisé s'il existe
    final_title = (("\n\n" + custom_title) if len(custom_title) else "")
    
    # 2. Ligne principale : Assets, Lag et Tolerance
    header = f"#Assets = {NbAssets}, "
    if position_lag_value:
        abs_val = abs(position_lag_value)
        header += f"position {'lagged' if position_lag_value > 0 else 'shifted forward'} by {abs_val} "
        header += f"{position_lag_units if abs_val > 1 else position_lag_units[:-1]}"
        if tolerance_value is not None:
            header += f", searching with tolerance on {tolerance_method} {tolerance_value} "
            header += f"{tolerance_units if tolerance_value > 1 else tolerance_units[:-1]}"
        header += ", "
        
    final_title += header + title_duration
    
    # 3. Seconde ligne conditionnelle : Coûts, Compounding et Risk-Free Rate
    second_line_parts = []
    
    if transaction_costs_bps > 0:
        second_line_parts.append(
            f"Transaction costs per Trade: {transaction_costs_bps} bps \u2192 Transaction yearly costs: {total_tc_yearly_pct:.2f} %."
        )
        
    if compound_returns:
        second_line_parts.append("Returns are compounded.")
        
    if risk_free_rate_pct != 0.0:
        sharpe_old_str = f"{total_sharpe_old:.2f}"
        sharpe_new_str = f"{total_sharpe_new:.2f}"
        second_line_parts.append(
            f"RF rate = {risk_free_rate_pct} % \u21D2 sharpe: {sharpe_old_str} \u2192 {sharpe_new_str}."
        )
        
    # S'il y a au moins un élément pour la 2ème ligne, on saute une ligne (\n) et on assemble avec un espace
    if second_line_parts:
        final_title += "\n" + " ".join(second_line_parts)
        
    return final_title

def compute_backtest_time(
    backtest_active_slice: slice, all_timestamps: pd.DatetimeIndex
) -> tuple[str, float, float, pd.DatetimeIndex]:
    """Calcule les informations temporelles sur la durée du backtest."""
    # Sélection de la période active
    active_timestamps = all_timestamps[backtest_active_slice]
    start_timestamp, end_timestamp = active_timestamps[0], active_timestamps[-1]

    # Calcul de la durée
    bt_nb_years, ymdhms_tuple = get_duration_details(end_timestamp, start_timestamp)
    duration_details_str = format_timedelta_str(*ymdhms_tuple)

    # Création du titre
    title_duration = (
        f"between {start_timestamp.strftime('%d/%m/%Y %H:%M:%S')} "
        f"and {end_timestamp.strftime('%d/%m/%Y %H:%M:%S')} ({duration_details_str})"
    )

    total_seconds = (end_timestamp - start_timestamp).total_seconds()

    return title_duration, bt_nb_years, total_seconds, active_timestamps

# Fonction principale pour calculer les métriques de backtest
def compute_backtest_metrics(
    timestamps: pd.DatetimeIndex,
    perf_names: np.ndarray,
    pos: np.ndarray,
    perf: np.ndarray,
    bt_nb_years: float,
    avg_nb_of_seconds_per_period: float,
    avg_nb_of_seconds_per_trading_day: float,
    detail_LS: bool = False,
    compute_max_DD: bool = True,
    compute_DD_points: bool = True,
    sort_by: Union[str, None] = 'sharpe',
    sort_ascending: bool = False,
    transaction_costs_bps: float = 0.0,
    compound_returns: bool = False,
    risk_free_rate_pct: float = 0.0
) -> tuple:
    """Calcule les métriques de performance détaillées du backtest."""

    #####################################################################
    # 1. Extraction des dimensions et des ordonnanceurs
    #####################################################################

    metrics_cols = [
        "category", "sharpe", "PnL/Trade", "holding period",
        "return (yearly)", "$\\sigma$ PnL (yearly)",
        "max DD", "relative max DD (in $\\sigma$s)", "%time position"
    ]

    sort_col: int = -1

    if type(sort_by) == str:
        if sort_by == "max DD" and compute_max_DD:
            sort_col = 6
        elif sort_by == "relative max DD ( in $\\sigma$s)" and compute_DD_points:
            sort_col = 7
        elif sort_by in metrics_cols[1:]:
            sort_col = metrics_cols[1:].index(sort_by) + 1

    BT_nb_periods, NbAssets = pos.shape
    tc_rate = max(0, transaction_costs_bps) * 1e-4
    rf_rate_decimal = risk_free_rate_pct / 100.0

    ######################################################################################################
    # SLT arranger: arrangement permettant de passer d'une présentation par L/S:
    #           SHORT         |           LONG          |     SHORT  +  LONG    |          TOTAL        |
    # Asset 1 | Asset 2 | ... | Asset 1 | Asset 2 | ... | Asset1 | Asset2 | ... | Asset1 | Asset2 | ... |
    #
    # à une présentation par Asset:
    #         Asset 1      |        Asset 2       |         TOTAL        |
    # Short | Long | S + L | Short | Long | S + L | Short | Long | S + L |
    ######################################################################################################

    if detail_LS:
        NbAssets_x_2: int = 2 * NbAssets
        NbAssets_x_3: int = 3 * NbAssets
        adjusted_NbAssets: int = NbAssets_x_3 + 3

        res_index: np.ndarray = np.concatenate((np.lib.stride_tricks.as_strided(perf_names, shape=(3, NbAssets),
                                                strides=(0,) + perf_names.strides).ravel(),
                                                ["Total", "Total", "Total"]))

        #####################################################################
        # 2. Construction des positions Short, Long consolidées
        #####################################################################

        position = np.full((BT_nb_periods, adjusted_NbAssets), 0., dtype=float)

        position[:, :NbAssets] = pos
        position[:, NbAssets: -3] = np.tile(pos, (1, 2))

        long_pos_mask = pos > 0
        position[:, :NbAssets][long_pos_mask] = 0
        position[:, -3] = position[:, :NbAssets].sum(axis=1)

        position[:, NbAssets: NbAssets_x_2][~long_pos_mask] = 0
        position[:, -2] = position[:, NbAssets: NbAssets_x_2].sum(axis=1)

        position[:, -1] = position[:, NbAssets_x_2: -3].sum(axis=1)

        abs_position = abs(position)
        abs_position[:, -3] = abs_position[:, :NbAssets].sum(axis=1)
        abs_position[:, -2] = abs_position[:, NbAssets: NbAssets_x_2].sum(axis=1)
        abs_position[:, -1] = abs_position[:, NbAssets_x_2: -3].sum(axis=1)

        #####################################################################
        # 3. Série du Turnover
        #####################################################################

        turnover = np.full((BT_nb_periods, adjusted_NbAssets), 0., dtype=float)
        turnover[0] = abs_position[0]
        if BT_nb_periods > 1:
            turnover[1:] = abs(np.diff(position, axis=0))
        turnover[-1] += abs_position[-1]

        turnover[:, -3] = turnover[:, :NbAssets].sum(axis=1)
        turnover[:, -2] = turnover[:, NbAssets: NbAssets_x_2].sum(axis=1)
        turnover[:, -1] = turnover[:, NbAssets_x_2: -3].sum(axis=1)

        #####################################################################
        # 4. Calcul du PnL
        #####################################################################

        pnl = np.full_like(position, 0., dtype=float)

        pnl[:, :-3] = position[:, :-3] * np.lib.stride_tricks.as_strided(perf, shape=(3,) + perf.shape[::-1],
                                                                         strides=(0,) + perf.strides[::-1]).reshape(
            NbAssets_x_3, -1).T

        pnl[:, -3:] = np.column_stack([pnl[:, :NbAssets].sum(axis=1),
                                       pnl[:, NbAssets: NbAssets_x_2].sum(axis=1),
                                       pnl[:, NbAssets_x_2: -3].sum(axis=1)])

        res_data: np.ndarray = np.full((adjusted_NbAssets, 9), "", dtype=object)

        ###################################################################
        # Réorganisation de l'ordre des colonnes
        ###################################################################

        SLT: np.ndarray = np.array(["Short", "Long", "Total"], dtype=str)
        res_data[:, 0] = np.concatenate((np.lib.stride_tricks.as_strided(SLT, shape=(3, NbAssets),
                                                                         strides=SLT.strides + (0,)).ravel(),
                                         ["Short", "Long", "Total"]))

    else:
        if NbAssets == 1:

            adjusted_NbAssets: int = 1
            res_index: np.ndarray = perf_names

            #####################################################################
            # 2. Construction des positions
            #####################################################################
            position = pos
            abs_position = abs(position)

            #####################################################################
            # 3. Série au Turnover
            #####################################################################
            turnover = np.full((BT_nb_periods, 1), 0., dtype=float)

            turnover[0] = abs_position[0]
            if BT_nb_periods > 1:
                turnover[1:] = abs(np.diff(position, axis=0))
            turnover[-1] += abs_position[-1]

            #####################################################################
            # 4. Calcul du PnL
            #####################################################################
            pnl = position * perf

        else:

            adjusted_NbAssets: int = NbAssets + 1
            res_index: np.ndarray = np.append(perf_names, "Total")

            #####################################################################
            # 2. Construction des positions
            #####################################################################
            position = np.full((BT_nb_periods, adjusted_NbAssets), 0., dtype=float)

            position[:, :-1] = pos
            position[:, -1] = pos.sum(axis=1)

            abs_position = abs(position)
            abs_position[:, -1] = abs_position[:, :-1].sum(axis=1)

            #####################################################################
            # 3. Série du Turnover
            #####################################################################
            turnover = np.full((BT_nb_periods, adjusted_NbAssets), 0., dtype=float)

            turnover[0] = abs_position[0]
            if BT_nb_periods > 1:
                turnover[1:] = abs(np.diff(position, axis=0))
            turnover[-1] += abs_position[-1]

            turnover[:, -1] = turnover[:, : -1].sum(axis=1)

            #####################################################################
            # 4. Calcul du PnL
            #####################################################################
            pnl = np.full_like(position, 0., dtype=float)

            pnl[:, :-1] = position[:, :-1] * perf
            pnl[:, -1] = pnl[:, :-1].sum(axis=1)

        res_data: np.ndarray = np.full((adjusted_NbAssets, 8), "", dtype=object)

    cum_turnover = turnover.sum(axis=0)

    #####################################################################
    # Déduction des frais de transaction et calcul du cumul
    #####################################################################
    if tc_rate > 0:
        pnl -= turnover * tc_rate
        total_tc_yearly_pct = (cum_turnover[-1] * tc_rate) / bt_nb_years * 100
    else:
        total_tc_yearly_pct = 0.0

    pnl_sum = pnl.sum(axis=0)

    if compound_returns:
        pnl_cumsum = np.cumprod(1 + pnl, axis=0) - 1
    else:
        pnl_cumsum = pnl.cumsum(axis=0)

    pnl_last_cumsum = pnl_cumsum[-1]

    #####################################################################
    # 5. Calcul des métriques de performance
    #####################################################################
    empty_assets_zero_result_row = np.full(adjusted_NbAssets, 0., dtype=float)

    # 5.a. %time position
    ###################################################################
    non_zero_position_mask = abs_position.astype(bool)
    sum_non_zero_position_mask = non_zero_position_mask.sum(axis=0)

    metrics_data = np.full((9, adjusted_NbAssets), np.nan, dtype=float)
    metrics_data[-1] = sum_non_zero_position_mask / BT_nb_periods

    # 5.b. Holding period
    ###################################################################
    holding_period_seconds = empty_assets_zero_result_row.copy()
    any_non_zero_position_mask = sum_non_zero_position_mask.astype(bool)

    cum_turnover_any_non_zero_position_mask = cum_turnover[any_non_zero_position_mask]
    holding_period_seconds[any_non_zero_position_mask] = abs_position[:, any_non_zero_position_mask].sum(axis=0) \
                                                         * 2 * avg_nb_of_seconds_per_period / cum_turnover_any_non_zero_position_mask

    metrics_data[-6] = holding_period_seconds

    # 5.c.PnL / Trade
    ###################################################################
    ret_per_trade = empty_assets_zero_result_row.copy()
    
    # On utilise la somme arithmétique nette (pnl_sum) pour la marge moyenne par trade
    pnl_sum_any_non_zero_position_mask = pnl_sum[any_non_zero_position_mask]
    ret_per_trade[any_non_zero_position_mask] = pnl_sum_any_non_zero_position_mask / cum_turnover_any_non_zero_position_mask
    metrics_data[-7] = ret_per_trade

    # 5.d. normed return (yearly)
    ###################################################################
    avg_ret = empty_assets_zero_result_row.copy()
    if compound_returns:
        avg_ret[any_non_zero_position_mask] = (1 + pnl_last_cumsum[any_non_zero_position_mask]) ** (1 / bt_nb_years) - 1
    else:
        avg_ret[any_non_zero_position_mask] = pnl_last_cumsum[any_non_zero_position_mask] / bt_nb_years

    metrics_data[-5] = avg_ret

    # 5.e.normed vol(/max pos /year)
    ###################################################################
    pnl_vol = pnl.std(axis=0, ddof=1) * np.sqrt(BT_nb_periods / bt_nb_years)
    metrics_data[-4] = pnl_vol

    # 5.f.sharpe(yearly)
    ###################################################################
    non_zero_vol_mask = pnl_vol.astype(bool)
    arithmetic_yearly_ret = pnl_sum / bt_nb_years
    
    # Calcul des deux Sharpe (avec et sans RF rate) pour l'affichage dynamique
    sharpe_yearly_without_rfr = np.zeros(adjusted_NbAssets, dtype=float)
    sharpe_yearly_without_rfr[non_zero_vol_mask] = arithmetic_yearly_ret[non_zero_vol_mask] / pnl_vol[non_zero_vol_mask]
    
    sharpe_yearly = np.zeros(adjusted_NbAssets, dtype=float)
    sharpe_yearly[non_zero_vol_mask] = (arithmetic_yearly_ret[non_zero_vol_mask] - rf_rate_decimal) / pnl_vol[non_zero_vol_mask]

    metrics_data[-8] = sharpe_yearly
    
    # On isole les Sharpe totaux pour le titre
    total_sharpe_old = sharpe_yearly_without_rfr[-1]
    total_sharpe_new = sharpe_yearly[-1]

    # 5.g. max DrawDown absolu
    ###################################################################
    if compute_max_DD:
        if compound_returns:
            peak = np.maximum.accumulate(1 + pnl_cumsum, axis=0)
            current_DD = 1.0 - (1 + pnl_cumsum) / np.maximum(peak, 1e-10)
        else:
            current_DD = np.maximum.accumulate(pnl_cumsum, axis=0) - pnl_cumsum
            
        end_DD = np.argmax(current_DD, axis=0)
        max_DD = -np.array([current_DD[i, j] for (j, i) in enumerate(end_DD)])

        metrics_data[-3] = max_DD
        res_data[:, -3] = [rd_unit(x, digits=4) for x in max_DD]

        # 5.h. max DrawDown relatif
        ###################################################################
        max_DD_rel = empty_assets_zero_result_row.copy()
        max_DD_rel[non_zero_vol_mask] = max_DD[non_zero_vol_mask] / pnl_vol[non_zero_vol_mask]

        metrics_data[-2] = max_DD_rel
        res_data[:, -2] = [rd_unit(x) for x in max_DD_rel]

    if compute_DD_points:

        where_has_DD = np.where(end_DD > 0)[0]
        start_DD = empty_assets_zero_result_row.astype(int)

        DDpoints = np.tile(np.full((2, 2), None), (adjusted_NbAssets, 1, 1))

        # 5.i.points des max DrawDown
        ###################################################################
        for whd in where_has_DD:
            current_end_DD = end_DD[whd]
            current_start_DD = np.argmax(pnl_cumsum[: current_end_DD + 1, whd], axis=0)

            start_DD[whd] = current_start_DD
            DDpoints[whd] = [[timestamps[current_start_DD], timestamps[current_end_DD]],
                             [pnl_cumsum[current_start_DD, whd], pnl_cumsum[current_end_DD, whd]]]

    else:
        DDpoints = None

    if sort_col:
        if detail_LS:
            metrics_data_col_total = metrics_data[sort_col, NbAssets_x_2: -3]
            metrics_data_col_total_argsort = np.append(np.argsort(metrics_data_col_total),
                                                       NbAssets) if sort_ascending else np.append(
                np.argsort(metrics_data_col_total)[::-1], NbAssets)

            repeated_metrics_argsort = np.lib.stride_tricks.as_strided(metrics_data_col_total_argsort,
                                                                       shape=metrics_data_col_total_argsort.shape + (
                                                                       3,),
                                                                       strides=metrics_data_col_total_argsort.strides + (
                                                                       0,)).ravel()

            array_012 = np.array([0, 1, 2], dtype=int)
            repeated_array_012 = np.lib.stride_tricks.as_strided(array_012, shape=(NbAssets + 1, 3),
                                                                 strides=(0,) + array_012.strides).ravel()

            metrics_argsort = 3 * repeated_metrics_argsort + repeated_array_012

        else:
            if NbAssets == 1:
                metrics_argsort = np.array([0])
            else:
                metrics_data_col_total = metrics_data[sort_col, :-1]
                metrics_argsort = np.append(np.argsort(metrics_data_col_total), NbAssets) if sort_ascending else np.append(
                    np.argsort(metrics_data_col_total)[::-1], NbAssets)

    else:
        metrics_argsort = None

    # Mise en forme des métriques pour affichage en str dans l'array res_data
    res_data[:, -8] = [str(np.round(x, 2)) for x in sharpe_yearly]  # sharpe annualisé
    res_data[:, -7] = [rd_bps(x) for x in ret_per_trade]  # PnL / Trade(bps)
    res_data[:, -6] = [format_days_timedelta(*seconds_to_dhms(x, nb_seconds_per_day=avg_nb_of_seconds_per_trading_day), days_subst="trading ") for x in holding_period_seconds]  # Holding period
    res_data[:, -5] = [rd_pct(x) for x in avg_ret]  # return total
    res_data[:, -4] = [rd_pct(x, digits=2) for x in pnl_vol]  # volatilité du PnL
    res_data[:, -1] = [rd_pct(x) for x in metrics_data[-1]]  # % temps en position

    res_Matrix = DataMatrix(index=res_index, columns=np.array(metrics_cols) if detail_LS else metrics_cols[1:],
                            values=res_data.astype(str))

    return position, pnl_cumsum, DDpoints, res_Matrix, metrics_argsort, total_tc_yearly_pct, total_sharpe_old, total_sharpe_new


# Fonction pour exécuter un backtest
def compute_backtest(
        pos_df: Union[pd.DataFrame, pd.Series],
        perf_df: Union[pd.DataFrame, pd.Series],
        transaction_costs_bps: float = 0.0,
        compound_returns: bool = False,
        risk_free_rate_pct: float = 0.0,
        position_lag_value: int = 1,
        position_lag_unit: str = 'units',
        tolerance_method: str = 'previous',
        tolerance_value: Optional[int] = None,
        tolerance_unit: str = 'minutes',
        position_fill_value: Optional[float] = None,
        perf_transform: bool = False,
        perf_fill_value: Optional[float] = 0,
        detail_LS: bool = False,
        detail_assets: bool = False,
        show_total: bool = True,
        compute_max_DD: bool = True,
        show_pnl: bool = True,
        show_position: bool = False,
        compute_DD_points: bool = True,
        title: str = '',
        check_index: bool = False,
        sort_by: Optional[str] = 'sharpe',
        sort_ascending: bool = False,
        display_nb_timestamps: int = 2560,
        improve_subsampled_position_display: bool = False,
        improve_subsampled_position_additional_timestamps_multiplier: float = 2.0) -> Backtest:
    """
    Exécute un backtest complet d'une stratégie de trading.
    """
    #####################################################################
    ################### I. VERIFICATION DES DONNEES
    #####################################################################
    if isinstance(pos_df, pd.Series):
        pos_df = pos_df.to_frame()

    if isinstance(perf_df, pd.Series):
        perf_df = perf_df.to_frame()

    if not len(pos_df) or not len(perf_df):
        return Backtest()

    assert isinstance(position_lag_value, (int, np.integer)), "position_lag_value should be an int"
    assert type(position_lag_unit) == str, "position_lag_unit should be a str"

    assert position_lag_unit in ['units', 'years', 'months', 'bdays', 'days', 'caldays', 'hours', 'minutes',
                                 'seconds'], 'provided position_lag_unit: ' + str(position_lag_unit) + (
        ' should be in [\'units\',\'years\','
        '\'months\', \'b.days\', \'days\', \'caldays\', \'hours\' , \'minutes\', \'seconds\']')

    assert tolerance_method in ['previous', 'exact', 'nearest', 'next'], 'provided tolerance_method: ' + \
                                                                          str(tolerance_method) + (
                                                                              ' should be in [\'previous\', \'exact\' , \'nearest\', \'next\'])')

    tolerance_value_has_type_int = isinstance(tolerance_value, (int, np.integer))

    if tolerance_value_has_type_int:
        assert tolerance_value >= 0, "position_lag_value_value should be >= 0"

    else:
        assert tolerance_value == None, "tolerance_value should be either an int( >= 0) or None"

    assert type(tolerance_unit) == str, "tolerance_unit should be a str"
    assert tolerance_unit in ['days', 'hours', 'minutes', 'seconds'], "provided position_unit: " + str(tolerance_unit) + \
                                                                      " tolerance_unit should be in [\'days\', \'hours\', \'minutes\', \'seconds\']"

    if (position_fill_value != None) and (type(position_fill_value) != float):
        try:
            position_fill_value = float(position_fill_value)
        except:
            raise TypeError("position_fill_value has wrong type: " + str(type(position_fill_value)) + \
                            ", should be either float castable or None")

    if (perf_fill_value) != None and (type(perf_fill_value) != float):
        try:
            perf_fill_value = float(perf_fill_value)
        except:
            raise TypeError("perf_fill_value has wrong type: " + str(type(perf_fill_value)) + \
                            " should be either float castable or None")

    assert pos_df.shape[1:] == perf_df.shape[1:], "dimension mismatch: position data has shape: " + str(pos_df.shape) + \
                                                  ", whereas performance data has shape : " + str(perf_df.shape)

    if type(sort_by) == str:
        assert sort_by in ("sharpe", "PnL/Trade", "holding period", "return (yearly)",
                           "$\\sigma$ PnL (yearly)", "max DD", "relative max DD (in $\\sigma$s)", "%time position"), \
            "if specified, sort_by parameter should be chosen among: \'sharpe\', \'PnL/Trade\', \'holding period\', \
                \'return (yearly)\', \'$\\sigma$ PnL (yearly)\', \
                \'max DD\', \'relative max DD (in $\\sigma$s)\', \'%time position\'"
    else:
        assert sort_by == None, "sort_by parameter: " + str(sort_by) + \
                                " should be either None(no sorting of metrics) or an str"


    #########################################################################################################

    # Si on n'a pas demandé le calcul du max DD alors on ne calcule pas non plus les points de départ/arrivée
    if not compute_max_DD:
        compute_DD_points = False

    NbAssets = perf_df.shape[1]
    timestamps = perf_df.index

    # On réindexe les positions sur les performances si les index sont différents ET check_index est True
    if check_index:
        if (len(pos_df) != len(perf_df)) or (pos_df.index != perf_df.index).any():
            warnings.warn("Les index de pos_df et perf_df ne sont pas alignés. pos_df a été réindexé sur perf_df avec un ffill pour éviter tout data leak.")
            pos_df = pos_df.reindex(timestamps, method='ffill')
            
    assert (pos_df.index == perf_df.index).all(), "Index mismatch between positions and performances. Set check_index=True to fix this automatically."

    timestamps_is_DatetimeIndex: bool

    if type(timestamps) == pd.core.indexes.datetimes.DatetimeIndex:
        dates = timestamps.date
        timestamps_is_DatetimeIndex = True

    else:
        dates = np.array([x.date() for x in timestamps])
        timestamps_is_DatetimeIndex = False

    perf_asset_names = perf_df.columns.values

    ##################################################################################################
    ################################ II. PREPARATION DES DONNEES
    ##################################################################################################

    ################################ II. a. Perf transform = prix-> perf
    ##################################################################################################

    # Si on a demandé la conversion des prix en perfs
    if perf_transform:
        perf_df = perf_df.pct_change()

    ################################ II.b . position lag
    ##################################################################################################
    # Si on doit appliquer un lag sur les positions
    if position_lag_value:

        native_position_lag_unit = (position_lag_unit == "units")

        # Cas oû position_lag_unit == 'units'
        if native_position_lag_unit or position_lag_unit == "days":

            pos = pos_df.values.copy()

            if native_position_lag_unit:
                if position_lag_value > 0:
                    pos[position_lag_value:] = pos[:-position_lag_value]
                    pos[: position_lag_value] = np.nan

                else:
                    pos[: position_lag_value] = pos[-position_lag_value:]
                    pos[position_lag_value:] = np.nan

            else:
                unique_dates, unique_dates_inv_indices = np.unique(dates, return_inverse=True)

                timestamps_mask = np.full(len(unique_dates), True, dtype=bool)

                if position_lag_value > 0:
                    timestamps_mask[-position_lag_value:] = False
                else:
                    timestamps_mask[position_lag_value:] = False

                len_lagged_pos = timestamps_mask[unique_dates_inv_indices].sum()

                if position_lag_value > 0:
                    new_pos = np.empty_like(pos)
                    new_pos[-len_lagged_pos:] = pos[timestamps_mask[unique_dates_inv_indices]]
                    new_pos[:-len_lagged_pos] = np.nan
                    pos = new_pos

                else:
                    new_pos = np.empty_like(pos)
                    new_pos[:len_lagged_pos] = pos[timestamps_mask[unique_dates_inv_indices]]
                    new_pos[len_lagged_pos:] = np.nan
                    pos = new_pos

        # Cas où position_lag_unit != "units"
        else:
            if position_lag_unit == 'bdays':
                date_offset = pd.tseries.offsets.BusinessDay(position_lag_value)
            else:
                if position_lag_unit == 'caldays':
                    position_lag_unit = 'days'

                date_offset = pd.DateOffset(**{position_lag_unit: position_lag_value})

            # La clé "exact": None a été ajoutée pour éviter un KeyError
            tolerance_compute_method_dict = {"previous": "ffill", "exact": None, "nearest": "nearest", "next": "bfill", None: None}
            tol_method = tolerance_compute_method_dict[tolerance_method]
            tolerance = str(tolerance_value) + ' ' + tolerance_unit if (tolerance_value != None) else None

            pos = pos_df.reindex(timestamps - date_offset, method=tol_method,
                                 tolerance=tolerance, axis=0).values

    else:
        pos = pos_df.values.copy()

    ################################ II .c. Merge Pos / Perf
    ##################################################################################################

    filtered_data = np.empty((len(timestamps), NbAssets * 2), dtype=float)

    filtered_data[:, :NbAssets] = pos
    filtered_data[:, NbAssets:] = perf_df.values

    infinite_data_mask = np.isinf(filtered_data)

    has_infinite_data_col = infinite_data_mask.any(axis=0)
    has_infinite_pos = has_infinite_data_col[: NbAssets].any()
    has_infinite_perf = has_infinite_data_col[NbAssets:].any()
    has_infinite_data = has_infinite_pos or has_infinite_perf

    infinite_or_nan_data_mask = infinite_data_mask | np.isnan(filtered_data)

    has_infinite_or_nan_data_row = infinite_or_nan_data_mask.any(axis=1)
    has_infinite_or_nan_data = has_infinite_or_nan_data_row.any()
    is_finite_index = np.where(~has_infinite_or_nan_data_row)[0]
    filtered_nb_points = len(is_finite_index)

    # Si il existe des données valides
    if filtered_nb_points:

        # Alors on extrait l'index de début du Backtest (premier 1'encontrê non NaN pour les 2)
        # et l'index de fin du Backtest (dernier rencontré non NaN pour les 2)
        start_index, end_index = is_finite_index[[0, -1]]

        # Masque de la plage de dates où le Bactest est •actif' (on pu calculer au moins 1 PnL et les positions
        # et les perfs n'ont pas encore disparues

        Backtest_active_slice = slice(start_index, end_index + 1)

        pos_perf = filtered_data[Backtest_active_slice]

        ################################ II.d . position fill et perf fill missing values
        ##################################################################################################
        if has_infinite_or_nan_data:
            if perf_fill_value == position_fill_value:

                if position_fill_value == None:
                    if has_infinite_data:
                        pos_perf[infinite_data_mask[Backtest_active_slice]] = 0.

                    pos_perf = pd.DataFrame(pos_perf).ffill().values

                else:
                    pos_perf[infinite_or_nan_data_mask[Backtest_active_slice]] = position_fill_value

            else:
                if position_fill_value == None:
                    if has_infinite_pos:
                        pos_perf[:, :NbAssets][infinite_data_mask[Backtest_active_slice, : NbAssets]] = 0.

                    pos_perf[:, :NbAssets] = pd.DataFrame(pos_perf[:, :NbAssets]).ffill().values
                    pos_perf[:, NbAssets:][
                        infinite_or_nan_data_mask[Backtest_active_slice, NbAssets:]] = perf_fill_value

                elif perf_fill_value == None:
                    pos_perf[:, : NbAssets][
                        infinite_or_nan_data_mask[Backtest_active_slice, :NbAssets]] = position_fill_value

                    if has_infinite_perf:
                        pos_perf[:, NbAssets:][infinite_data_mask[Backtest_active_slice, NbAssets:]] = 0.

                    pos_perf[:, NbAssets:] = pd.DataFrame(pos_perf[:, NbAssets:]).ffill().values

                else:
                    active_infinite_or_nan_data_mask = infinite_or_nan_data_mask[Backtest_active_slice]
                    pos_perf[:, :NbAssets][active_infinite_or_nan_data_mask[:, :NbAssets]] = position_fill_value
                    pos_perf[:, NbAssets:][active_infinite_or_nan_data_mask[:, NbAssets:]] = perf_fill_value

        ##################################################################################################
        ################################ III. MESURE DU TEMPS (APPROCHE MEDIANE)
        ##################################################################################################

        title_duration, BT_nb_years, _, active_timestamps = \
            compute_backtest_time(Backtest_active_slice, timestamps)

        diff_timestamps = np.diff(active_timestamps.values).astype('timedelta64[s]').astype(float)
        median_step_seconds = np.median(diff_timestamps) if len(diff_timestamps) > 0 else 86400.0

        nb_active_periods = end_index - start_index + 1

        if timestamps_is_DatetimeIndex:
            active_dates = active_timestamps.date
        else:
            active_dates = np.array([x.date() for x in active_timestamps])

        nb_unique_days = len(np.unique(active_dates))

        avg_periods_per_day = nb_active_periods / nb_unique_days
        avg_nb_of_seconds_per_trading_day = avg_periods_per_day * median_step_seconds
        avg_nb_of_seconds_per_period = median_step_seconds

        ##################################################################################################
        ################################ IV. CALCUL DES METRIQUES DE PERFORMANCE
        ##################################################################################################

        Backtest_positions_values, Backtest_pnl_cumsum_values, Backtest_max_DD_dots, Backtest_metrics_DataMatrix, \
            Backtest_metrics_argsort, total_tc_yearly_pct, total_sharpe_old, total_sharpe_new = \
            compute_backtest_metrics(active_timestamps, perf_asset_names, pos_perf[:, :NbAssets],
                                     pos_perf[:, NbAssets:], BT_nb_years, avg_nb_of_seconds_per_period,
                                     avg_nb_of_seconds_per_trading_day, detail_LS, compute_max_DD, compute_DD_points,
                                     sort_by if detail_assets else None, sort_ascending,
                                     transaction_costs_bps, compound_returns, risk_free_rate_pct)

        # ------------------------------------------------------------------------------------------------
        # NOUVELLE GÉNÉRATION DU TITRE REFACTORISÉE
        # ------------------------------------------------------------------------------------------------
        Backtest_title = build_backtest_title(
            custom_title=title,
            NbAssets=NbAssets,
            position_lag_value=position_lag_value,
            position_lag_units=position_lag_unit,
            tolerance_method=tolerance_method,
            tolerance_value=tolerance_value,
            tolerance_units=tolerance_unit,
            title_duration=title_duration,
            transaction_costs_bps=transaction_costs_bps,
            total_tc_yearly_pct=total_tc_yearly_pct,
            compound_returns=compound_returns,
            risk_free_rate_pct=risk_free_rate_pct,
            total_sharpe_old=total_sharpe_old,
            total_sharpe_new=total_sharpe_new
        )

        BT_dict = {}

        if detail_LS:
            output_perf_names = np.lib.stride_tricks.as_strided(perf_asset_names, shape=(3, NbAssets),
                                                                strides=(0,) + perf_asset_names.strides).ravel()

            Short_Long = np.array([" - Short", " - Long", ""])
            repeated_Short_Long = np.lib.stride_tricks.as_strided(
                Short_Long, shape=(3, NbAssets), strides=Short_Long.strides + (0,)).ravel()

            output_perf_names = np.append([a + b for a, b in zip(output_perf_names, repeated_Short_Long)],
                                          ["Total - Short", "Total - Long", "Total"])

            NbAssets_x_3 = NbAssets * 3
            Asset_arranger = np.concatenate(
                (np.arange(NbAssets_x_3).reshape(3, -1).T.ravel(), np.arange(NbAssets_x_3, NbAssets_x_3 + 3)))

            if detail_assets:
                output_slice = slice(None, None if show_total else - 3, None)
                if type(Backtest_metrics_argsort) == np.ndarray:
                    sliced_Asset_arranger = (Asset_arranger[Backtest_metrics_argsort])[output_slice]
                else:
                    sliced_Asset_arranger = output_slice
            else:
                sliced_Asset_arranger = slice(-3, None if show_total else - 1, None)

        else:
            output_perf_names = np.append(perf_asset_names, "Total")
            if detail_assets:
                if show_total or NbAssets == 1:
                    output_slice = slice(None, None, None)
                else:
                    output_slice = slice(None, -1, None)

                if type(Backtest_metrics_argsort) == np.ndarray:
                    sliced_Asset_arranger = Backtest_metrics_argsort[output_slice]
                else:
                    sliced_Asset_arranger = output_slice
            else:
                sliced_Asset_arranger = slice(-1, None, None)

        # ------------------------------------------------------------------------------------------------
        # V. LOGIQUE DE SOUS-ECHANTILLONNAGE AVANCÉ
        # ------------------------------------------------------------------------------------------------
        total_timestamps = len(active_timestamps)
        
        if display_nb_timestamps > 0 and total_timestamps > display_nb_timestamps:
            # Base : sous-échantillonnage régulier pour assurer l'intégrité globale de la courbe PnL
            idx_array = np.unique(np.round(np.linspace(0, total_timestamps - 1, display_nb_timestamps)).astype(int))
            
            # AMÉLIORATION DES POSITIONS : Repêchage ciblé des trades manquants (Turnover-based)
            if show_position and improve_subsampled_position_display and improve_subsampled_position_additional_timestamps_multiplier > 0:
                
                # Récupération des positions cibles (celles qui seront affichées)
                target_positions = Backtest_positions_values[:, sliced_Asset_arranger]
                
                # Calcul du "turnover" absolu à chaque timestep pour repérer les trades
                pos_diff = np.zeros(total_timestamps)
                pos_diff[1:] = np.sum(np.abs(np.diff(target_positions, axis=0)), axis=1)
                pos_diff[0] = np.sum(np.abs(target_positions[0]))
                
                # Index où un trade a eu lieu
                change_indices = np.where(pos_diff > 0)[0]
                
                # Index des trades qui sont passés à la trappe lors du linspace
                missing_changes = np.setdiff1d(change_indices, idx_array)
                
                if len(missing_changes) > 0:
                    # Budget de points supplémentaires autorisés
                    budget = int(display_nb_timestamps * improve_subsampled_position_additional_timestamps_multiplier)
                    
                    missing_turnovers = pos_diff[missing_changes]
                    
                    # Bruit aléatoire ultra-rapide pour départager équitablement les ex-aequo
                    random_tie_breaker = np.random.rand(len(missing_turnovers))
                    
                    # np.lexsort trie de manière ascendante. 
                    # On utilise -missing_turnovers comme dernière clé pour trier par turnover décroissant.
                    sort_order = np.lexsort((random_tie_breaker, -missing_turnovers))
                    
                    # Sélection des meilleurs candidats manquants selon le budget
                    top_missing_indices = missing_changes[sort_order][:budget]
                    
                    # Injection des repêchés dans l'axe de temps
                    idx_array = np.unique(np.concatenate((idx_array, top_missing_indices)))
            
            # SÉCURITÉ MAX DD : Injection des points de Max DD dans le masque
            if compute_DD_points and (Backtest_max_DD_dots is not None):
                dd_ts = []
                for mdd in Backtest_max_DD_dots[sliced_Asset_arranger]:
                    if mdd is not None and mdd[0] is not None:
                        if mdd[0][0] is not None:
                            dd_ts.extend(mdd[0])
                
                if dd_ts:
                    # pd.Index().isin garantit la conversion sécurisée entre Timestamp et datetime64
                    idx_dd = np.where(pd.Index(active_timestamps).isin(dd_ts))[0]
                    idx_array = np.unique(np.concatenate((idx_array, idx_dd)))

            # Application du masque final
            disp_timestamps = active_timestamps[idx_array]
            disp_pnl = Backtest_pnl_cumsum_values[idx_array]
            if show_position:
                disp_pos = Backtest_positions_values[idx_array]
        else:
            disp_timestamps = active_timestamps
            disp_pnl = Backtest_pnl_cumsum_values
            if show_position:
                disp_pos = Backtest_positions_values

        if show_pnl:
            BT_dict["pnl_cumsum"] = DataMatrix(index=disp_timestamps.to_numpy(),
                                               columns=output_perf_names[sliced_Asset_arranger],
                                               values=disp_pnl[:, sliced_Asset_arranger])

        if compute_DD_points:
            BT_dict["max_DD"] = Backtest_max_DD_dots[sliced_Asset_arranger]

        if show_position:
            BT_dict["position"] = disp_pos[:, sliced_Asset_arranger]

        return Backtest(nb_assets=NbAssets, title=Backtest_title,
                        metrics=Backtest_metrics_DataMatrix[sliced_Asset_arranger], bt_dict=BT_dict)

    else:
        return Backtest()

# Fonction pour tracer les résultats d'un Backtest
def plot_backtest(
        bt: Backtest,
        fig_width: float = 20,
        fig_height: float = 15,
        font_size: float = 10,
        table_row_height: float = 0.4,
        title_vertical_space: float = 0.0,
        dpi: Optional[int] = 72,
        show_fig: bool = True,
        save_fig: bool = False,
        save_path: Optional[str] = None,
        default_save_extension: str = ".png",
        return_fig = False,
        display_nb_timestamps: int = -1):
    """
    Génère une visualisation graphique complète des résultats d'un backtest financier.
    Cette fonction trace le PnL cumulé et, optionnellement, les positions et les
    points de drawdown maximum. Elle inclut également une table des métriques
    de performance du backtest si disponibles.

    Parameters
    ----------
    bt : Backtest
        L'objet Backtest contenant toutes les données et métriques à visualiser.
        Il doit inclure `bt.bt_dict` avec au moins 'pnl_cumsum' et optionnellement
        'position' et 'max_DD', et `bt.metrics` pour la table des métriques.
    fig_width : float, optional
        La largeur de la figure en pouces, par défaut 20.
    fig_height: float, optional
        La hauteur de la figure en pouces, par défaut 15.
    font_size : float, optional
        La taille de police de base utilisée pour les éléments de texte, par défaut 10.
    table_row_height : float, optional
        La hauteur de chaque ligne dans le tableau des métriques, par défaut 0.4.
    title_vertical_space : float, optional
        Un ajustement vertical pour la position du titre de la figure, par défaut 0.0.
        Une valeur positive déplace le titre vers le haut.
    dpi : Optional[int], optional
        La résolution de la figure en points par pouce (dots per inch).
        Utilisé lors de la sauvegarde de l'image. Par défaut 72.
    show_fig : bool, optional
        Si True, affiche la figure générée. Si False, la figure n'est pas affichée
        mais peut être sauvegardée. Par défaut True.
    save_fig : bool, optional
        Si True, la figure est sauvegardée sur le disque. Par défaut False.
    return_fig : bool, optional
        Si True, retourne le tuple (fig, ax) pour pouvoir manipuler la figure.
        Si False, ne retourne rien. Par défaut False.
    save_path : Optional[str], optional
        Le chemin complet (y compris le nom du fichier) où la figure doit être sauvegardée.
        Si None et `save_fig` est True, le fichier est sauvegardé sous "Backtest.png"
        dans le répertoire courant. Par défaut None.
    default_save_extension : str, optional
        L'extension par défaut à utiliser si `save_path` est spécifié sans extension
        lorsque `save_fig` est True. Par défaut ".png".
    display_nb_timestamps : int, optional
        Nombre maximum de points (timestamps) affichés sur les graphes pour plus de fluidité.
        Si <= 0, aucun sous-échantillonnage n'est effectué par cette fonction (-1 par défaut 
        pour respecter le travail pré-calculé par compute_Backtest).
    """

    BT_dict = bt.bt_dict

    if BT_dict:
        title = bt.title
        metrics = bt.metrics

        has_metrics = metrics is not None

        if has_metrics and ("pnl_cumsum" in BT_dict):
            nb_rows: int = len(metrics.index)
        else:
            nb_rows = 0

        table_height = table_row_height * (1 + nb_rows)

        # --- HEURISTIQUES DE POLICE ET WRAP ---
        table_fontsize = max(8, min(font_size, fig_width * 0.8))
        legend_fontsize = max(8, table_fontsize - 1)
        title_fontsize = table_fontsize + 2

        if title:
            # Nettoyage pour éviter les \n\n invisibles (générés par compute_Backtest) qui faussent le centrage
            clean_title = title.strip()
            
            # 1 pouce = 72 points. Un caractère prend environ 60% de sa taille de police en largeur.
            max_chars_per_line = int((fig_width * 72) / (title_fontsize * 0.6))
            
            # Prise en compte des \n manuels introduits par l'utilisateur
            final_title = '\n'.join([textwrap.fill(line, width=max(30, max_chars_per_line)) for line in clean_title.split('\n')])
            nb_rows_title = final_title.count('\n') + 1
        else:
            nb_rows_title = 0
            final_title = ""

        total_fig_height = table_height + fig_height

        if not show_fig:
            plt.ioff()

        fig = plt.figure(figsize=(fig_width, total_fig_height), dpi=dpi)

        # Ajout d'un gridspec pour un meilleur contrôle
        gs = fig.add_gridspec(
            nrows=2, ncols=1,
            height_ratios=[table_height, fig_height],
            hspace=0.02
        ) if has_metrics else fig.add_gridspec(nrows=1, ncols=1)

        ax = fig.add_subplot(gs[-1])
        if has_metrics:
            ax3 = fig.add_subplot(gs[0])
            ax3.axis("off")
            # INDISPENSABLE : Empêche tight_layout d'écraser la largeur du graphique à cause du tableau !
            ax3.set_in_layout(False)  

        # --- DOWNSAMPLING LOGIC POUR L'AFFICHAGE ---
        pnl_cumsum_raw = BT_dict["pnl_cumsum"]
        total_timestamps = len(pnl_cumsum_raw.index)

        if display_nb_timestamps > 0 and total_timestamps > display_nb_timestamps:
            idx_array = np.unique(np.round(np.linspace(0, total_timestamps - 1, display_nb_timestamps)).astype(int))
            
            # Sécurité : Injecter les points de Max DD dans le masque pour éviter qu'ils ne flottent
            if "max_DD" in BT_dict:
                dd_ts = []
                for mdd in BT_dict["max_DD"]:
                    if mdd is not None and mdd[0] is not None and mdd[0][0] is not None:
                        dd_ts.extend(mdd[0])
                if dd_ts:
                    # Utilisation de pd.Index pour garantir la compatibilité des types (Timestamp vs datetime64)
                    idx_dd = np.where(pd.Index(pnl_cumsum_raw.index).isin(dd_ts))[0]
                    idx_array = np.unique(np.concatenate((idx_array, idx_dd)))

            pnl_cumsum = pnl_cumsum_raw[idx_array]
            if "position" in BT_dict:
                positions_plot = BT_dict["position"][idx_array]
            else:
                positions_plot = None
        else:
            pnl_cumsum = pnl_cumsum_raw
            if "position" in BT_dict:
                positions_plot = BT_dict["position"]
            else:
                positions_plot = None
        # ---------------------------------------------

        indices, columns, values = pnl_cumsum.index, pnl_cumsum.columns, pnl_cumsum.values
        columns_list = list(columns)

        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlabel("DateTime", loc="center", fontsize=10)

        ax.tick_params(axis="y", color="dodgerblue", labelsize=10, labelcolor="dodgerblue")
        ax.set_ylabel("cumulated PnL", loc="center", c="dodgerblue", fontsize=10)

        # 1. Tracé du PnL en premier pour générer les objets Line2D et leurs couleurs
        lines = ax.plot(indices, values, zorder=2)

        # 2. Gestion de l'épaisseur des lignes en fonction du nom (Total vs Short/Long)
        for line, col_name in zip(lines, columns):
            if " - Short" in str(col_name) or " - Long" in str(col_name):
                line.set_linewidth(1.0)
                line.set_alpha(0.6)
            else:
                line.set_linewidth(1.5)
                line.set_alpha(1.0)

        # 3. Tracé des positions sur l'axe secondaire
        if positions_plot is not None:
            ax2 = ax.twinx()
            ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax2.tick_params(axis="y", color="gray", labelsize=10, labelcolor="gray")
            ax2.set_ylabel("position", loc="center", c="gray", fontsize=10)
            ax2.grid(True, which="both", axis="y", color="gray", alpha=0.2, linestyle="--")

            positions = positions_plot
            for j, col_name in enumerate(columns):
                color = lines[j].get_color()
                pos_values = positions[:, j]
                col_name_str = str(col_name)
                
                if " - Short" in col_name_str or " - Long" in col_name_str:
                    # Ligne fine pour les Short/Long pour préserver la hiérarchie visuelle
                    ax2.plot(indices, pos_values, drawstyle="steps-post", color=color, linewidth=1.0, alpha=0.6, zorder=3)
                    
                    # On applique le fill_between SI la ligne Total associée est absente (ex: show_total=False)
                    base_name = col_name_str.replace(" - Short", "").replace(" - Long", "")
                    if base_name not in columns_list:
                        ax2.fill_between(indices, 0, pos_values, step="post", color=color, alpha=0.10, zorder=3)
                else:
                    # Fill between systématique pour les lignes Total
                    ax2.fill_between(indices, 0, pos_values, step="post", color=color, alpha=0.15, zorder=3)
                    ax2.plot(indices, pos_values, drawstyle="steps-post", color=color, linewidth=1.5, alpha=0.8, zorder=3)

            ax2.relim()
            ax2.autoscale()

        # 4. Légende : Mise en gras dynamique si ligne principale (linewidth > 1.0)
        ax_legend = ax.legend(lines, columns, fontsize=legend_fontsize, title="cumulated PnL",
                              title_fontsize=legend_fontsize + 2, labelcolor="linecolor", loc="upper left")

        for orig_line, leg_text in zip(lines, ax_legend.get_texts()):
            if orig_line.get_linewidth() > 1.0:
                leg_text.set_weight("bold")

        ax_legend.zorder = 0
        ax_legend.get_title().set_color("dodgerblue")
        ax.add_artist(ax_legend)

        ax.relim()
        ax.autoscale()
        ax.grid(True, which="both", axis="y", color="dodgerblue", alpha=0.3)
        ax.grid(True, which="both", axis="x", alpha=0.5)

        if "max_DD" in BT_dict:
            max_DDs = BT_dict["max_DD"]
            ax.plot([], [], 'rx', mew=1.5, ms=8, label="max DD")
            ax.legend(fontsize=legend_fontsize, loc="lower right")

            for max_DD in max_DDs:
                ax.plot(*max_DD, 'rx', mew=2, ms=12, label="max DD", zorder=1)

        if has_metrics:
            metrics_nb_cols = len(metrics.columns)
            
            # Utilisation exclusive de `lines` pour récupérer les couleurs
            rowcolours = np.array([line.get_color() for line in lines])
            cellcolours = np.lib.stride_tricks.as_strided(
                rowcolours,
                shape=(nb_rows, metrics_nb_cols),
                strides=rowcolours.strides + (0,))

            pair_colors = np.array(["#FF4500", "#01153E"], dtype=str)
            tiled_pair_colors = np.lib.stride_tricks.as_strided(
                pair_colors, shape=(4, 2), strides=(0,) + pair_colors.strides).reshape(8)

            is_Total_row = np.array([False, True])

            if metrics_nb_cols == 9:
                colcolours = np.append(["#808080"], tiled_pair_colors)
                has_category_col: bool = True
                if nb_rows > 1:
                    is_Total_row = np.append(False, metrics.values[:, 0] == "Total")
            else:
                colcolours = tiled_pair_colors
                has_category_col = False
                if nb_rows > 1:
                    is_Total_row = np.append(False, metrics.index == "Total")

            # Bbox ajusté pour un padding parfaitement symétrique
            tab = ax3.table(
                cellText=metrics.values, rowLabels=metrics.index, colLabels=metrics.columns,
                loc="center", cellLoc="center", cellColours=cellcolours, rowColours=rowcolours,
                colColours=colcolours, alpha=0.5,
                bbox=[0.01, 0, 0.98, 1])

            tab.set_fontsize(table_fontsize)
            tab.auto_set_column_width(col=list(range(metrics_nb_cols)))
            tab.scale(4, 5 * table_row_height)

            for (row, col), cell in tab.get_celld().items():
                alpha = 1
                weight = "normal"

                if is_Total_row[row]:
                    weight = "bold"
                    alpha = 0.8 if col > 0 and col % 2 else 0.9
                elif row == 0 or col < 0:
                    weight = "bold"
                elif col > 0:
                    if col % 2:
                        alpha -= 0.2

                cell.set_text_props(color="w", weight=weight)
                cell.set_alpha(alpha)

            _, u_index, u_counts = np.unique(
                metrics.index, return_index=True, return_counts=True)

            adjacent_indices_list = [
                list(range(start_index, start_index + count + 1))
                for start_index, count in zip(u_index, u_counts - 1)
                if count]

            for adjacent_indices in adjacent_indices_list:
                adjusted_nb_adjacent_indices = len(adjacent_indices) - 2
                edges = ["TRL"] + ["RL"] * adjusted_nb_adjacent_indices + ["BRL"]
                texts = []

                for ind, edge in zip(adjacent_indices, edges):
                    cell = tab[ind + 1, -1]
                    cell.set_text_props(color="black")
                    cell.visible_edges = edge
                    local_text = cell.get_text()
                    texts.append(local_text)

                for local_text in (texts[:1] + texts[-1:] if adjusted_nb_adjacent_indices else texts[:1]):
                    local_text.set_visible(False)

        # --- GÉOMÉTRIE ABSOLUE ET PARFAITEMENT SYMÉTRIQUE DU TITRE ---
        # 1. Le layout ne regarde que le graphe (garantit la largeur MAXIMALE et CONSTANTE)
        fig.tight_layout() 
        
        if final_title:
            # Marges de coussin en pouces (Haut et Bas)
            pad_inches = 0.15
            
            # Hauteur réelle du texte (interligne 1.4 pour engloutir les majuscules et accents)
            text_height_inches = (title_fontsize * 1.4 * max(1, nb_rows_title)) / 72.0 
            
            # La "bande" totale requise = Marge + Texte + Marge
            title_band_inches = (2 * pad_inches) + text_height_inches
            title_band_ratio = title_band_inches / total_fig_height
            
            # On repousse le graphe/tableau vers le bas
            top_margin = 1.0 - title_band_ratio + title_vertical_space
            fig.subplots_adjust(top=max(0.70, min(0.98, top_margin)))
            
            # On place l'ancrage EXACTEMENT AU MILIEU de la bande
            # (Adieu le déséquilibre optique causé par va='top')
            y_text_center = 1.0 - (title_band_ratio / 2.0) + title_vertical_space
            fig.suptitle(final_title, fontsize=title_fontsize, y=y_text_center, va='center')
        else:
            fig.subplots_adjust(top=0.98)
        # -------------------------------------------------------------

        if save_fig:
            if save_path is None:
                save_path = Path().absolute().joinpath("Backtest.png")
            else:
                if "." not in save_path:
                    save_path += default_save_extension

                if "\\" not in save_path:
                    save_path = Path().absolute().joinpath(save_path)

            plt.savefig(save_path)

        if not show_fig:
            plt.close(fig)

        plt.ion()

    if return_fig:
        return fig, ax