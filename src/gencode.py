# =======================================================
# Generates the source code for the pNCCL wrappers that
# are used to trace NCCL functions.
# This is achieved in a similar fashion as in
# liballprof.
# 
# Siyuan Shen - ETH Zurich
# =======================================================
import re
import os
import argparse
from typing import List, Dict, Tuple, TextIO

# Mapping from NCCL data types to the corresponding format string
type_to_format = {
    "int": "%d",
    "float": "%f",
    "double": "%lf",
    "size_t": "%zu",
    "ncclDataType_t": "%d",
    "ncclScalarResidence_t": "%p",
    "ncclComm_t": "%p",
    "cudaStream_t": "%p",
    "ncclUniqueId": "%p",
    "ncclRedOp_t": "%d",
}



class NcclProfCodegen:
    def __init__(self, nccl_header: str,
                 output_path: str) -> None:
        self.nccl_header = nccl_header
        self.output_path = output_path


    def __write_prolog(self, output_file: TextIO, verbose: bool) -> None:
        """
        Writes the prolog of the generated C file.
        """
        # Opens the prolog file and writes it to the output file
        with open("prolog.c", "r") as prolog_file:
            output_file.write(prolog_file.read())
        output_file.write("\n\n")
        if verbose:
            print("[INFO] Prolog written to the output file.")
    
    def __write_func_wrapper(self, output_file: TextIO, operation_name: str,
                             arguments: str, verbose: bool) -> None:
            """
            Writes the wrapper function for the given NCCL operation.
            """
            if verbose:
                print(f"[INFO] Writing wrapper for {operation_name}.")
            # Writes the function signature
            output_file.write(f"// Wrapper for {operation_name}\n")
            output_file.write(f"ncclResult_t {operation_name}({arguments})\n")
            output_file.write("{\n")
            # Splits the arguments by comma
            arguments = arguments.split(",")
            # Writes the function body
            output_file.write("  ncclResult_t nccl_result;\n")
            output_file.write("  double start_time = get_time();\n")
            pnccl_arg_names = ", ".join([(args.split(" ")[-1]).lstrip("*") for args in arguments])
            output_file.write(f"  nccl_result = p{operation_name}({pnccl_arg_names});\n")
            output_file.write("  double end_time = get_time();\n")

            # Traces the function
            output_file.write(f'  WRITE_TRACE("{operation_name}:')
            
            format_args = [("%f", "start_time")]
            for arg in arguments:
                arg_tokens = arg.split()
                if not arg_tokens:
                    continue

                assert len(arg_tokens) == 2 or len(arg_tokens) == 3, f"Invalid argument {arg}"
                if len(arg_tokens) == 2:
                    arg_type, arg_name = arg_tokens
                else:
                    # Const modifier
                    arg_type = arg_tokens[1]
                    arg_name = arg_tokens[2]
                
                is_ptr = arg_type.endswith("*") or arg_name.startswith("*")
                is_double_ptr = arg_type.endswith("**") or arg_name.startswith("**")
                arg_name = arg_name.lstrip("*")

                # FIXME Redundant code, but I think it's clearer this way
                if is_double_ptr:
                    format_args.append(("%lx", f"(uintptr_t) *{arg_name}"))
                    continue
                
                if is_ptr:
                    format_args.append(("%lx", f"(uintptr_t) {arg_name}"))
                    continue
                    
                format_str = type_to_format.get(arg_type)
                if format_str is None:
                    raise ValueError(f"Unknown type {arg_type}")
                if format_str == "%p":
                    format_args.append(("%lx", f"&{arg_name}"))
                else:
                    format_args.append((format_str, arg_name))
                
            format_args.append(("%f", "end_time"))

            format_strs, arg_names = zip(*format_args)
            output_file.write(":".join(format_strs) + '\\n", ' + ", ".join(arg_names) + ");\n")
            output_file.write("  return nccl_result;\n")
            output_file.write("}")

            output_file.write("\n\n")

    def generate_wrapper(self, verbose: bool = False) -> None:
        """
        Generates all the code for the C wrapper as per the header file.
        The results will be written to `output`.
        """
        # Makes sure that the header file exists
        if not os.path.exists(self.nccl_header):
            raise FileNotFoundError(f"Header file {self.nccl_header} does not exist.")
        # Opens the output file
        output_file = open(self.output_path, "w")
        # Writes the prolog to the output file
        self.__write_prolog(output_file, verbose)

        # Reads all the content of the header file
        header_file = open(self.nccl_header, "r")

        # Finds all the NCCL operations with the pattern
        # "ncclResult_t\s+(nccl.+)\((.*)\)"
        nccl_operations = re.findall(r"ncclResult_t\s+(nccl.+)\(([^;]*)\)", header_file.read())
        # Writes the wrapper code for each operation
        for operation in nccl_operations:
            # Extracts the operation name and the arguments
            operation_name = operation[0]
            arguments = operation[1]
            self.__write_func_wrapper(output_file, operation_name,
                                      arguments, verbose)
        header_file.close()

        output_file.close()
        if verbose:
            print(f"[INFO] Wrapper code written to {self.output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="NCCL tracer codegen",
        description="Generates wrapper for NCCL functions present in the NCCL header file.")

    parser.add_argument("--header", required=False, dest="nccl_header",
                        default="nccl.h",
                        help="Path to the NCCL header file [default: nccl.h]")
    parser.add_argument("-o", "--output", required=False, dest="output_path",
                        default="nccl_wrapper.c",
                        help="Path to the generated C wrapper file [default: nccl_wrapper.c]")
    parser.add_argument("-v", "--verbose", required=False, dest="verbose",
                        action="store_true", default=False,
                        help="Enables verbose output during code generation.")

    args = parser.parse_args()
    
    codegen = NcclProfCodegen(args.nccl_header, args.output_path)

    codegen.generate_wrapper(args.verbose)

