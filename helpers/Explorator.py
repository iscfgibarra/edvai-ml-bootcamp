import pandas as pd
import numpy as np


class Explorator:

    def __init__(self, data_raw):
        self._data = self.to_dataframe(data_raw)

    def to_dataframe(self, data_raw):
        if isinstance(data_raw, list):
            data_raw = np.array(data_raw)

        if (len(data_raw.shape)) > 2:
            raise Exception("No puedo manejar mas de 2 dimensiones")

        if isinstance(data_raw, pd.Series):
            data_aux = pd.DataFrame({data_raw.name: data_raw})
        elif isinstance(data_raw, np.ndarray):
            if data_raw.shape == 1:
                data_aux = pd.DataFrame({'var': data_raw}).convert_dtypes()
            else:
                data_aux = pd.DataFrame(data_raw).convert_dtypes()
        else:
            data_aux = data_raw

        return data_aux

    def totals(self):
        tot_rows = len(self._data)

        d2 = self._data.isnull().sum().reset_index()
        d2.columns = ['variable', 'qty_nan']

        d2[['perc_nan']] = d2[['qty_nan']] / tot_rows

        d2['qty_zeros'] = (self._data == 0).sum().values

        d2['perc_zeros'] = d2[['qty_zeros']] / tot_rows

        d2['unique'] = self._data.nunique().values

        d2['type'] = [str(x) for x in self._data.dtypes.values]

        return d2

    def numerical_vars(self, exclude_var=None):
        num_v = self._data.select_dtypes(include=['int64', 'float64']).columns
        if exclude_var is not None:
            num_v = num_v.drop(exclude_var)
        return num_v

    def categorical_vars(self, exclude_var=None):
        cat_v = self._data.select_dtypes(include=['object', 'category', 'string']).columns
        if exclude_var is not None:
            cat_v = cat_v.drop(exclude_var)
        return cat_v

    def numerical_profile(self):
        d = self._data[self.numerical_vars()]

        des1 = pd.DataFrame({'mean': d.mean().transpose(),
                             'std_dev': d.std().transpose()})

        des1['variation_coef'] = des1['std_dev'] / des1['mean']

        d_quant = d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().add_prefix('p_')

        des2 = des1.join(d_quant, how='outer')

        des_final = des2.copy()

        des_final['variable'] = des_final.index

        des_final = des_final.reset_index(drop=True)

        des_final = des_final[
            ['variable', 'mean', 'std_dev', 'variation_coef', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95',
             'p_0.99']]

        return des_final

    def _frequency(self, var, name):
        cnt = var.value_counts()
        df_res = pd.DataFrame({'frequency': var.value_counts(), 'percentage': var.value_counts() / len(var)})
        df_res.reset_index(drop=True)

        df_res[name] = df_res.index

        df_res = df_res.reset_index(drop=True)

        df_res['cumulative_perc'] = df_res.percentage.cumsum() / df_res.percentage.sum()

        df_res = df_res[[name, 'frequency', 'percentage', 'cumulative_perc']]

        return df_res

    def frequency(self):
        cat_v = self.categorical_vars()
        if len(cat_v) == 0:
            return 'No hay variables categoricas.'

        if len(cat_v) > 1:
            for col in cat_v:
                print(self._frequency(self._data[col], name=col))
                print('\n----------------------------------------------------------------\n')
        else:
            col = cat_v[0]
            return self._frequency(self._data[col], name=col)

    def calc_correlation(self, method='pearson'):
        d_cor = self._data.corr(method)
        d_cor2 = d_cor.reset_index()
        d_long = d_cor2.melt(id_vars='index')
        d_long.columns = ['v1', 'v2', 'R']
        d_long[['R2']] = d_long[['R']] ** 2
        d_long2 = d_long.query("v1 != v2")
        return d_long2
