"""
Calculator Tool for Educational Content System
Provides simple mathematical calculation capabilities
"""

import re
import math
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class CalculationResult:
    """Result of a mathematical calculation"""
    expression: str
    result: Union[int, float, str]
    success: bool
    error_message: Optional[str] = None


class CalculatorTool:
    """Simple mathematical calculation tool"""
    
    def __init__(self):
        """Initialize the calculator tool"""
        self.basic_operations = ['+', '-', '*', '/', '%']
        self.advanced_operations = ['^', '**', 'sqrt', 'log', 'sin', 'cos', 'tan', 'abs']
        self.math_constants = {'pi': math.pi, 'e': math.e}
    
    def calculate(self, expression: str) -> CalculationResult:
        """
        Calculate the result of a mathematical expression
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            CalculationResult object
        """
        try:
            # Clean and validate expression
            clean_expr = self._clean_expression(expression)
            
            if not clean_expr:
                return CalculationResult(
                    expression=expression,
                    result="",
                    success=False,
                    error_message="Empty or invalid expression"
                )
            
            # Replace mathematical constants
            processed_expr = self._replace_constants(clean_expr)
            
            # Calculate result
            result = self._evaluate_expression(processed_expr)
            
            return CalculationResult(
                expression=expression,
                result=result,
                success=True
            )
            
        except Exception as e:
            return CalculationResult(
                expression=expression,
                result="",
                success=False,
                error_message=str(e)
            )
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize the mathematical expression"""
        if not expression:
            return ""
        
        # Remove extra whitespace
        expr = expression.strip()
        
        # Replace common symbols
        expr = expr.replace('×', '*').replace('÷', '/')
        expr = expr.replace('**', '^')  # Handle power notation
        
        return expr
    
    def _replace_constants(self, expression: str) -> str:
        """Replace mathematical constants with their values"""
        expr = expression
        for const_name, const_value in self.math_constants.items():
            expr = expr.replace(const_name, str(const_value))
        return expr
    
    def _evaluate_expression(self, expression: str) -> Union[int, float]:
        """Evaluate the mathematical expression safely"""
        # Handle power operations
        expr = expression.replace('^', '**')
        
        # Handle square root
        expr = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expr)
        
        # Handle other math functions
        expr = re.sub(r'log\(([^)]+)\)', r'math.log10(\1)', expr)
        expr = re.sub(r'sin\(([^)]+)\)', r'math.sin(\1)', expr)
        expr = re.sub(r'cos\(([^)]+)\)', r'math.cos(\1)', expr)
        expr = re.sub(r'tan\(([^)]+)\)', r'math.tan(\1)', expr)
        expr = re.sub(r'abs\(([^)]+)\)', r'abs(\1)', expr)
        
        # Evaluate the expression
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        
        # Round to reasonable precision
        if isinstance(result, float):
            return round(result, 6)
        return result
    
    def get_supported_operations(self) -> Dict[str, Any]:
        """Get information about supported operations"""
        return {
            "basic": self.basic_operations,
            "advanced": self.advanced_operations,
            "constants": list(self.math_constants.keys()),
            "description": "Simple calculator supporting basic math operations, functions, and constants"
        }
    
    def validate_expression(self, expression: str) -> Dict[str, Any]:
        """Validate a mathematical expression"""
        try:
            # Clean the expression
            clean_expr = self._clean_expression(expression)
            
            if not clean_expr:
                return {
                    'valid': False,
                    'errors': ['Empty or invalid expression'],
                    'warnings': [],
                    'suggestions': ['Enter a valid mathematical expression']
                }
            
            # Try to evaluate it to check for syntax errors
            try:
                processed_expr = self._replace_constants(clean_expr)
                result = self._evaluate_expression(processed_expr)
                return {
                    'valid': True,
                    'errors': [],
                    'warnings': [],
                    'suggestions': []
                }
            except Exception as e:
                return {
                    'valid': False,
                    'errors': [str(e)],
                    'warnings': [],
                    'suggestions': ['Check for typos or unsupported operations']
                }
                
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}'],
                'warnings': [],
                'suggestions': []
            }
    
    def solve_equation(self, equation: str) -> CalculationResult:
        """
        Solve a simple linear equation
        
        Args:
            equation: Linear equation as string (e.g., "x + 5 = 10")
            
        Returns:
            CalculationResult object
        """
        try:
            if '=' not in equation:
                return CalculationResult(
                    expression=equation,
                    result="",
                    success=False,
                    error_message="Equation must contain '=' sign"
                )
            
            # Split the equation
            left, right = equation.split('=', 1)
            left = left.strip()
            right = right.strip()
            
            # Simple linear equation solver for x
            if 'x' not in left and 'x' not in right:
                return CalculationResult(
                    expression=equation,
                    result="",
                    success=False,
                    error_message="Equation must contain variable 'x'"
                )
            
            # Handle simple cases like "x + 5 = 10" or "x - 3 = 7"
            try:
                if 'x' in left and 'x' not in right:
                    # Format: x + 5 = 10
                    right_val = float(eval(self._replace_constants(right), {"__builtins__": {}}, {"math": math}))
                    
                    # Parse left side
                    if left == 'x':
                        result = right_val
                    elif '+' in left:
                        parts = left.split('+')
                        if len(parts) == 2:
                            if parts[0].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[1].strip()), {"__builtins__": {}}, {"math": math}))
                                result = right_val - constant
                            elif parts[1].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[0].strip()), {"__builtins__": {}}, {"math": math}))
                                result = right_val - constant
                            else:
                                raise ValueError("Unsupported equation format")
                        else:
                            raise ValueError("Unsupported equation format")
                    elif '-' in left:
                        parts = left.split('-')
                        if len(parts) == 2:
                            if parts[0].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[1].strip()), {"__builtins__": {}}, {"math": math}))
                                result = right_val + constant
                            else:
                                raise ValueError("Unsupported equation format")
                        else:
                            raise ValueError("Unsupported equation format")
                    else:
                        raise ValueError("Unsupported equation format")
                
                elif 'x' in right and 'x' not in left:
                    # Format: 10 = x + 5
                    left_val = float(eval(self._replace_constants(left), {"__builtins__": {}}, {"math": math}))
                    
                    # Parse right side
                    if right == 'x':
                        result = left_val
                    elif '+' in right:
                        parts = right.split('+')
                        if len(parts) == 2:
                            if parts[0].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[1].strip()), {"__builtins__": {}}, {"math": math}))
                                result = left_val - constant
                            elif parts[1].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[0].strip()), {"__builtins__": {}}, {"math": math}))
                                result = left_val - constant
                            else:
                                raise ValueError("Unsupported equation format")
                        else:
                            raise ValueError("Unsupported equation format")
                    elif '-' in right:
                        parts = right.split('-')
                        if len(parts) == 2:
                            if parts[0].strip() == 'x':
                                constant = float(eval(self._replace_constants(parts[1].strip()), {"__builtins__": {}}, {"math": math}))
                                result = left_val + constant
                            else:
                                raise ValueError("Unsupported equation format")
                        else:
                            raise ValueError("Unsupported equation format")
                    else:
                        raise ValueError("Unsupported equation format")
                else:
                    raise ValueError("Equation too complex for simple solver")
                
                # Round result
                if isinstance(result, float):
                    result = round(result, 6)
                
                return CalculationResult(
                    expression=equation,
                    result=f"x = {result}",
                    success=True
                )
                
            except Exception as e:
                return CalculationResult(
                    expression=equation,
                    result="",
                    success=False,
                    error_message=f"Could not solve equation: {str(e)}"
                )
                
        except Exception as e:
            return CalculationResult(
                expression=equation,
                result="",
                success=False,
                error_message=str(e)
            )
