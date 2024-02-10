import sys
import os

sys.path.append(os.path.abspath(".."))

from utils import libs_cpd

gaussian = libs_cpd.namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


def online_detection(time_series=list(), window=20, queue_window=10, treshold_coef=3):
    ''' 
        An online implementation of Kalman Filter Change Point Detection based on risiduals,
    and current time-series statistics for treshold updating.
    
    Note:
        queue_window - min possible distance between two predicted CPs.
        treshold_coef - if residuals above this treshold than we can detect CP. Based on np.meean + 3*np.std
        where 3 is the coef.
        window - the length of data which we save in memory.
    '''

    dp = [0] * window  # массив с предсказаниями по CPs
    queue_risiduals = [0] * window  # очередь с рядом остатков по которому отбивают трешхолд
    queue_cp = [0] * queue_window  # очередь CPs
    queue_ts = time_series[:window]  # р
    queue_gaussian = [gaussian(libs_cpd.np.mean(queue_ts), libs_cpd.np.var(queue_ts))] * 2

    for i in range(window, len(time_series)):  # Это цикл позволяет нам получать данные раз в 9 секунд условно.
        gaussian_likelihood = queue_gaussian[0]  # смотрим начальное состояние MEAN VAR
        gaussian_prior = gaussian(libs_cpd.np.mean(queue_ts),
                                  libs_cpd.np.var(queue_ts))  # смотрим обновление состояние MEAN VAR

        x = update(gaussian_prior, gaussian_likelihood)  # обновляем значение MEAN VAR
        queue_gaussian.pop(0)  # удаляем старое значение
        queue_gaussian.append(gaussian_prior)  # добавляем новое значение likelihoo

        next_prediction = libs_cpd.np.random.normal(
            loc=x.mean)  # Генерируем значение исходя из нового знания об gaussian
        next_value = time_series[i]  # П
        risidual_value = abs(next_prediction - next_value)

        mean_risidual_prev = libs_cpd.np.mean(queue_risiduals)
        std_risidual_prev = libs_cpd.np.std(queue_risiduals)
        queue_risiduals.pop(0)  # удаляем старое значение risiduals
        queue_risiduals.append(risidual_value)  # новое значение risiduals

        if risidual_value >= mean_risidual_prev + std_risidual_prev * treshold_coef:  # проверка что мы нашли переход
            if max(queue_cp) != 1:  # queue filter IF
                queue_cp.pop(0)
                queue_cp.append(1)
                dp.append(1)
            else:  # queue filter else
                queue_cp.pop(0)
                queue_cp.append(0)
                dp.append(0)
        else:
            dp.append(0)
            queue_cp.pop(0)
            queue_cp.append(0)

        queue_ts.pop(0)  # удаляем из очереди старое значение VALUE
        queue_ts.append(next_value)  # добавляем в очередь новое значение VALUE
    return dp
