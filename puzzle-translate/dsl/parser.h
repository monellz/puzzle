#ifndef __PUZZLE_PARSER_H
#define __PUZZLE_PARSER_H

namespace mlir::puzzle::dsl {
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

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
    Pad,
    Bound,
    Iter,
    Stencil,
    Kernel,
    RArrow,
    If,
    Assign,
    Comma,
    Colon,
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

using StackItem = std::variant<Token, std::unique_ptr<Module>, std::unique_ptr<Decl>,
                               std::vector<std::unique_ptr<Info>>, std::unique_ptr<Info>, std::unique_ptr<Block>,
                               std::vector<std::unique_ptr<Stmt>>, std::unique_ptr<Stmt>, std::vector<int64_t>,
                               std::unique_ptr<Expr>, std::vector<std::unique_ptr<Expr>>>;

struct Lexer {
  std::string_view string;
  std::string_view fn;
  u32 line, col;

  explicit Lexer(std::string_view string, std::string_view fn = "") : string(string), fn(fn), line(1), col(1) {}
  Token next();
};

struct Parser {
  std::variant<std::unique_ptr<Module>, Token> parse(Lexer &lexer);
};

} // namespace mlir::puzzle::dsl

#endif
