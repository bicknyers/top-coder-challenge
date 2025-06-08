import torch


def compile_to_raw_python(ckpt):
    index = 0
    run_str = "import math\n\n"
    run_str += "def calc(miles, rec, dur):\n"
    run_str += "\tvar_0_0 = miles\n"
    run_str += "\tvar_0_1 = rec\n"
    run_str += "\tvar_0_2 = dur\n"
    index += 1
    state_dict = ckpt["state_dict"]
    apply_norm = True
    for key in state_dict:
        val = state_dict[key]
        if 'norm' in key:
            if not apply_norm:
                continue
            # Normalize is a tad different from other layers
            if 'weight' in key:
                # Linear Weight
                hidden_index = -1
                for row in val:
                    hidden_index += 1
                    input_var = "var_" + str(index - 1) + "_" + str(hidden_index)
                    set_var = "var_" + str(index) + "_" + str(hidden_index) + "_a"
                    run_str += "\t" + set_var + " = " + input_var + " / " + str(float(row)) + '\n'
            elif 'bias' in key:
                # Linear Bias
                hidden_index = -1
                for row in val:
                    hidden_index += 1
                    input_var = "var_" + str(index) + "_" + str(hidden_index) + "_a"
                    set_var = "var_" + str(index) + "_" + str(hidden_index)
                    run_str += "\t" + set_var + " = " + input_var + " - " + str(float(row)) + '\n'
                index += 1
        elif 'weight' in key:
            # Linear Weight
            hidden_index = -1
            for row in val:
                hidden_index += 1
                for i in range(len(row)):
                    input_var = "var_" + str(index-1) + "_" + str(i)
                    set_var = "var_" + str(index) + "_" + str(i) + "_h" + str(hidden_index)
                    run_str += "\t" + set_var + " = " + input_var + " * " + str(float(row[i])) + '\n'
            for j in range(hidden_index+1):
                set_var = "var_" + str(index) + "_" + str(j) + "_a"
                run_str += "\t" + set_var + " = 0"
                for i in range(len(val[0])):
                    input_var = "var_" + str(index) + "_" + str(i) + "_h" + str(j)
                    run_str += " + " + input_var
                run_str += '\n'
        elif 'bias' in key:
            # Linear Bias
            hidden_index = -1
            for row in val:
                hidden_index += 1
                input_var = "var_" + str(index) + "_" + str(hidden_index) + "_a"
                set_var = "var_" + str(index) + "_" + str(hidden_index)
                run_str += "\t" + set_var + " = max(0.0, " + input_var + " + " + str(float(row)) + ')\n'
            index += 1

    if apply_norm:
        # run_str += "\tret_out = 1 / (1 + (math.e ** (-var_5_0)))\n"
        # run_str += "\treturn ret_out\n"
        run_str += "\treturn var_6_0\n"
    else:
        # run_str += "\tret_out = 1 / (1 + (math.e ** (-var_3_0)))\n"
        # run_str += "\treturn ret_out\n"
        run_str += "\treturn var_4_0\n"
    return run_str


version = 40
load_path = 'logs/lightning_logs/version_' + str(version) + '/checkpoints/epoch=29-step=930.ckpt'
ckpt = torch.load(load_path)


print(compile_to_raw_python(ckpt))

file_path = "generated.py"

with open(file_path, 'w') as file:
    file.write(compile_to_raw_python(ckpt))

# print('l1')
# print(ckpt['state_dict']['l1.weight'])
# print()
# print(ckpt['state_dict']['l1.bias'])
# print()
# print('l3')
# print(ckpt['state_dict']['l3.weight'])
# print()
# print(ckpt['state_dict']['l3.bias'])
