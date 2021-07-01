# @title
import numpy as np
import copy
import math
import time
from scipy.optimize import linprog


class Simplex:

    def __init__(self, A, B, C, key_k=0):
        self.key_k = key_k
        self.A = A
        self.B = B
        self.C = -C
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.primary_B = copy.copy(B)
        self.primary_n = self.n

    def get_basis_indices(self):
        identity_matrix = np.eye(min(self.n, self.m), dtype=int)
        self.basis_indices = []
        for j in range(self.m):
            identity_matrix_column = identity_matrix[:, j]
            for i in range(self.n):
                current_matrix_column = self.A[:, i]
                if (np.array_equal(identity_matrix_column, current_matrix_column)):
                    self.basis_indices.append(i)

    def get_c_basis(self):
        self.c_basis = []
        for k in self.basis_indices:
            c_basis_element = self.C[k]
            self.c_basis.append(c_basis_element)

    def add_basis(self):
        self.c_basis = []
        self.basis_indices = []
        identity_matrix = np.eye(self.m)
        self.A = np.hstack((self.A, identity_matrix))
        self.C = np.hstack((self.C, np.zeros(self.m)))
        for i in range(self.m):
            self.n = self.n + 1
            self.basis_indices.append(self.n - 1)
            self.c_basis.append(0)
        self.primary_basis_indices = copy.copy(self.basis_indices)

    def get_evaluation(self):
        self.evaluation_list = []
        evaluation_p0 = ((self.B * self.c_basis).sum() - 0)
        self.evaluation_list.append(evaluation_p0)
        for j in range(self.n):
            column_evaluation = (self.A[:, j] * self.c_basis).sum() - self.C[j]
            self.evaluation_list.append(column_evaluation)

    def get_theta(self, vector):
        theta = 999999999
        theta_index = 0
        if (self.key_k != 0):
            max_neg=0
            max_neg_ind=0
            for i in range(vector.size):
                if self.B[i] < 0 and max_neg<abs(self.B[i]):
                  max_neg=self.B[i]
                  max_neg_ind=i
            return max_neg_ind
        for i in range(vector.size):
            if ((vector[i] > 0) and (self.B[i] / vector[i] < theta)):
                theta = self.B[i] / vector[i]
                theta_index = i
        return theta_index

    def get_new_element(self):
        max_evaluation_index = np.argmax(self.evaluation_list[1:])
        vector = self.A[:, max_evaluation_index]
        theta_index_row = self.get_theta(vector)
        return theta_index_row, max_evaluation_index

    def recalculating_table(self):
        ind_row, ind_col = self.get_new_element()
        self.basis_indices[ind_row] = ind_col
        self.c_basis[ind_row] = self.C[ind_col]
        new_row = np.divide(self.A[ind_row, :], self.A[ind_row, ind_col])
        self.B[ind_row] = self.B[ind_row] / self.A[ind_row, ind_col]
        self.A[ind_row] = new_row
        for i in range(self.m):
            if i != ind_row:
                mnozh = (-self.A[i, ind_col]) / self.A[ind_row, ind_col]
                new_b = self.B[ind_row] * mnozh + self.B[i]
                new_row = np.multiply(self.A[ind_row], mnozh) + self.A[i]
                self.A[i] = new_row
                self.B[i] = new_b

    def set_x(self):
        x = np.zeros(self.n)
        for i in range(self.m):
            x[self.basis_indices[i]] = self.B[i]
        if self.key_k == 0:
            return x[:self.primary_n]
        else:
            return x[:(self.n - self.m)]

    def print_info(self, i):
        print("Итерация - ", i)
        print(self.A, " - A ")
        print("x - ", self.set_x())
        print("Целевая функция - ", -self.evaluation_list[0])
        print("-" * 33)

    def forward_simplex(self):
        self.add_basis()
        self.get_evaluation()
        i = 0
        while max(self.evaluation_list[1:]) > 0:
            self.print_info(i)
            i += 1
            self.recalculating_table()
            self.get_evaluation()
        print("Результат: ")
        self.print_info(i)
        return self.primary_basis_indices, self.A, self.m

    def forward_only_result(self):
        if self.key_k == 0:
            self.add_basis()
        else:
            self.get_basis_indices()
            self.get_c_basis()
        self.get_evaluation()
        while max(self.evaluation_list[1:]) > 0:
            self.recalculating_table()
            self.get_evaluation()
        return self.set_x(), -self.evaluation_list[0]


class SensAnalysis:
    def __init__(self, primary_basis_indices, res_A, m, primary_A, primary_B, primary_C, iteration_count_sa3=10):
        self.primary_basis_indices = primary_basis_indices
        self.res_A = res_A
        self.primary_A = primary_A
        self.primary_B = primary_B
        self.primary_C = primary_C
        self.m = m
        self.res_Basis = self.get_res_Basis()
        self.iteration_count_sa3 = iteration_count_sa3

    def get_res_Basis(self):
        res_Basis = []
        for i in self.primary_basis_indices:
            res_Basis.append((self.res_A[:, i]).tolist())
        res_Basis = np.transpose(np.array(res_Basis))
        return res_Basis

    def sens_analysis_1(self):
        delta_b = []
        for j in range(self.m):
            delta_b.append((self.res_Basis[j] * self.primary_B).sum())
        delta_b = np.array(delta_b)
        self.left_confines, self.right_confines = [], []
        for k in range(delta_b.size):
            left = self.res_Basis[:, k]
            right = delta_b
            right = -right
            mensh_ravno, bolsh_ravno = [], []
            for i in range(left.size):
                if (left[i] < 0):
                    mensh_ravno.append((right[i] / left[i]))
                elif (left[i] > 0):
                    bolsh_ravno.append(right[i] / left[i])
                else:
                    None
            self.normalization_inequality(mensh_ravno, bolsh_ravno, k)

    def normalization_inequality(self, mensh_ravno, bolsh_ravno, indx):
        if mensh_ravno:
            min_mensh_ravno = float('{:.3f}'.format(min(mensh_ravno) + self.primary_B[indx]))
        else:
            min_mensh_ravno = "не ограничено"
        if bolsh_ravno:
            max_bolsh_ravno = float('{:.3f}'.format(max(bolsh_ravno) + self.primary_B[indx]))
        else:
            max_bolsh_ravno = "не ограничено"

        if ((type(max_bolsh_ravno) is str or type(min_mensh_ravno) is str) or max_bolsh_ravno < min_mensh_ravno):
            self.left_confines.append(max_bolsh_ravno)
            self.right_confines.append(min_mensh_ravno)
            print(max_bolsh_ravno, "<=b", indx + 1, "<=", min_mensh_ravno)
        else:
            self.left_confines.append("не изменимо")
            self.right_confines.append("не изменимо")

    def get_max_values(self, simplex_result, vector_B_or_Coef, simplex_x, k):
        if simplex_result:
            max_value = max(simplex_result)
            max_result_index = simplex_result.index(max_value)
            B_or_Coef_with_max_value = vector_B_or_Coef[max_result_index]
            x_with_max_value = simplex_x[max_result_index]
            print("f(x) - ", max_value)
            print("x - ", x_with_max_value)
            if k == 2:
                print("B - ", B_or_Coef_with_max_value)
            elif k == 3:
                print("coef - ", B_or_Coef_with_max_value)
            else:
                None
        else:
            print("текущее решение не улучшить")

    def sens_analysis_2(self):
        simplex_x = []
        simplex_result = []
        vector_b = []
        for z in range(self.m):
            if (self.left_confines[z] != "не ограничено" and self.left_confines[z] != "не изменимо" and
                    self.right_confines[z] != "не ограничено" and self.right_confines[z] != "не изменимо"):
                for d in range(math.ceil(self.left_confines[z]), math.floor(self.right_confines[z])):
                    new_B = copy.copy(self.primary_B)
                    new_B[z] = d
                    primary_new_B = copy.copy(new_B)
                    smpl = Simplex(self.primary_A, new_B, self.primary_C)
                    x, y = smpl.forward_only_result()
                    simplex_x.append(x)
                    vector_b.append(primary_new_B.tolist())
                    simplex_result.append(y)

        self.get_max_values(simplex_result, vector_b, simplex_x, 2)

    def get_coefficient_b(self, new_B):
        test = []
        for j in range(self.m):
            test.append((self.res_Basis[j] * new_B).sum())

        return np.array(test)

    def sens_analysis_3(self):
        simplex_x = []
        simplex_result = []
        vector_coef = []

        print("~"*20, "Итерации","~"*20)
        for z in range(self.m):
            if (self.left_confines[z] != "не ограничено" and self.left_confines[z] != "не изменимо" and
                    self.right_confines[z] != "не ограничено" and self.right_confines[z] != "не изменимо"):
                for iteration in range(1, self.iteration_count_sa3):
                    new_B = copy.copy(self.primary_B)
                    new_B[z] = round(self.left_confines[z]) - iteration
                    new_B = self.get_coefficient_b(new_B)
                    smpl = Simplex(self.res_A, new_B, np.hstack((self.primary_C, np.zeros(self.m))), 1)
                    x, y = smpl.forward_only_result()
                    print(y ," при коэффициенте ", new_B)
                    simplex_x.append(x)
                    vector_coef.append(new_B.tolist())
                    simplex_result.append(y)
                for iteration in range(1, self.iteration_count_sa3):
                    new_B1 = copy.copy(self.primary_B)
                    new_B1[z] = round(self.right_confines[z]) + iteration
                    new_B1 = self.get_coefficient_b(new_B1)
                    smpl = Simplex(self.res_A, new_B1, np.hstack((self.primary_C, np.zeros(self.m))), 1)
                    x, y = smpl.forward_only_result()
                    print(y ," при коэффициенте ", new_B1)
                    simplex_x.append(x)
                    vector_coef.append(new_B1.tolist())
                    simplex_result.append(y)
        print("~"*40)
        print("Наилучший результат: ")
        self.get_max_values(simplex_result, vector_coef, simplex_x, 3)

    def forward_sens_analysis(self):
        print("Анализ на чувствительность: ")
        print("(1) Интервал допустимых значений ограничений запасов удобрений: ")
        self.sens_analysis_1()
        print("(2) Максимальное значение целевой функции при изменении ограничений в допустимом диапазоне: ")
        self.sens_analysis_2()
        print(
            "(3) Максимальное значение целевой функции при изменении ограничений за пределами допустимого: ")
        self.sens_analysis_3()


def get_data_from_json():
    import json
    with open('data.json') as json_file:
        data = json.load(json_file)
        a = data['a']
        b = data['b']
        c = data['c']
    return np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)


def comparision_with_linprog(A, B, C):
    A_in = copy.copy(A)
    B_in = copy.copy(B)
    C_in = copy.copy(C)

    print("~" * 22, "Linprog ", "~" * 22)
    start_time = time.time()
    res = linprog(-C_in, A_in, B_in)
    print(res.x, " - x")
    print(-res.fun, " - f(x)")
    print("Время выполнения", (time.time() - start_time))
    print("~" * 54, "\n")

    print("~" * 15, "Собственная реализация", "~" * 15)
    start_time = time.time()
    smpl = Simplex(A_in, B_in, C_in)
    x, y = smpl.forward_only_result()
    print(x, " - x")
    print(y, " - f(x)")
    print("Время выполнения", (time.time() - start_time))
    print("~" * 55, "\n")


if __name__ == '__main__':
    A, B, C = get_data_from_json()
    primary_A = copy.copy(A)
    primary_B = copy.copy(B)
    primary_C = copy.copy(C)
    comparision_with_linprog(A, B, C)

    smpl = Simplex(A, B, C)
    primary_basis_indices, res_A, m = smpl.forward_simplex()

    sens_a = SensAnalysis(primary_basis_indices, res_A, m, primary_A, primary_B, primary_C)
    sens_a.forward_sens_analysis()



