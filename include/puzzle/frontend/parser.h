#ifndef __PARSER_H
#define __PARSER_H

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "puzzle/frontend/ast.h"

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

// using namespace puzzle::ast;
namespace puzzle::ast {

struct Token {
  enum Kind : u32 {
    _Eps,
    _Eof,
    _Err,
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Ge,
    Gt,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Unary,
    RPar,
    Empty,
    Else,
    In,
    Out,
    Grid,
    Const,
    Pad,
    UpperBound,
    LowerBound,
    If,
    Stencil,
    Assign,
    Comma,
    Semi,
    Not,
    LPar,
    LBrk,
    RBrk,
    LBrc,
    RBrc,
    IntLit,
    FloatLit,
    Ident
  } kind;
  std::string_view piece;
  u32 line, col;
};

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

struct Token;

struct Lexer {
  std::string_view string;
  std::string_view fn;
  u32 line, col;

  explicit Lexer(std::string_view string, std::string_view fn = "") : string(string), fn(fn), line(1), col(1) {}
  Token next();
};

struct Parser {
  std::variant<std::unique_ptr<Module>, Token> parse(Lexer& lexer);
};

}  // namespace puzzle::ast

#endif
