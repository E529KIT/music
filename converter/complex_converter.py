def complex_list_to_double_list(complex_list):
    return [complex_number.real for complex_number in complex_list] + [complex_number.imag for complex_number in
                                                                       complex_list]


def double_list_to_complex_list(double_list):
    size = len(double_list) / 2
    return [complex(double_list[i], double_list[i + size]) for i in range(size)]


if __name__ == '__main__':
    complex_list = [complex(i, i * 100) for i in range(10)]
    print complex_list
    double_list = complex_list_to_double_list(complex_list)
    print double_list
    converted_complext_list = double_list_to_complex_list(double_list)
    print converted_complext_list
    print complex_list == converted_complext_list
