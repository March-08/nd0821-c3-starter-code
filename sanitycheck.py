import ast
import sys


class TestSanityChecker(ast.NodeVisitor):
    def __init__(self):
        self.get_test_found = False
        self.post_test_count = 0
        self.issues = []

    def visit_FunctionDef(self, node):
        # Check for GET test
        if "get" in node.name.lower():
            self.get_test_found = True
            self._check_assertions(node, node.name)

        # Check for POST test
        if "post" in node.name.lower():
            self.post_test_count += 1
            self._check_assertions(node, node.name)

    def _check_assertions(self, node, test_name):
        assertions = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        if not assertions:
            self.issues.append(f"No assertions found in {test_name}")

    def generate_report(self):
        report_lines = []
        if not self.get_test_found:
            report_lines.append("Issue: No GET test found.")

        if self.post_test_count == 0:
            report_lines.append("Issue: No POST test(s) found.")

        if self.issues:
            report_lines.append("Detailed Issues Detected with Your Test Cases:")
            for issue in self.issues:
                report_lines.append(f"- {issue}")
        else:
            report_lines.append(
                "No issues detected. Your test cases seem to be in good shape!"
            )

        return "\n".join(report_lines)


def main(test_file_path):
    with open(test_file_path, "r") as file:
        tree = ast.parse(file.read(), filename=test_file_path)

    checker = TestSanityChecker()
    checker.visit(tree)
    report_content = checker.generate_report()

    # Write the report to a file
    with open("report.txt", "w") as report_file:
        report_file.write(report_content)
    print("Report generated and saved to report.txt.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sanitycheck.py path/to/test_main.py")
    else:
        main(sys.argv[1])
