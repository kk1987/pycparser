import sys
from pycparser.c_ast import *
from pycparser.c_parser import CParser, Coord, ParseError
from pycparser.c_lexer import CLexer


def expand_decl(decl):
    """ Converts the declaration into a nested list.
    """
    typ = type(decl)
    
    if typ == TypeDecl:
        return ['TypeDecl', expand_decl(decl.type)]
    elif typ == IdentifierType:
        return ['IdentifierType', decl.names]
    elif typ == ID:
        return ['ID', decl.name]
    elif typ in [Struct, Union]:
        decls = [expand_decl(d) for d in decl.decls or []]
        return [typ.__name__, decl.name, decls]
    else:        
        nested = expand_decl(decl.type)
    
        if typ == Decl:
            if decl.quals:
                return ['Decl', decl.quals, decl.name, nested]
            else:
                return ['Decl', decl.name, nested]
        elif typ == Typename: # for function parameters
            if decl.quals:
                return ['Typename', decl.quals, nested]
            else:
                return ['Typename', nested]
        elif typ == ArrayDecl:
            dimval = decl.dim.value if decl.dim else ''
            return ['ArrayDecl', dimval, nested]
        elif typ == PtrDecl:
            return ['PtrDecl', nested]
        elif typ == Typedef:
            return ['Typedef', decl.name, nested]
        elif typ == FuncDecl:
            if decl.args:
                params = [expand_decl(param) for param in decl.args.params]
            else:
                params = []
            return ['FuncDecl', params, nested]

#-----------------------------------------------------------------
class NodeVisitor(object):
    def __init__(self):
        self.current_parent = None
        
    def visit(self, node):
        """ Visit a node. 
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def visit_FuncCall(self, node):
        print("Visiting FuncCall")
        print(node.show())
        print('---- parent ----')
        print(self.current_parent.show())
    
    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a 
            node. Implements preorder visiting of the node.
        """
        oldparent = self.current_parent
        self.current_parent = node
        for c in node.children():
            self.visit(c)
        self.current_parent = oldparent


if __name__ == "__main__":    
    source_code = '''
    int a;
'''


    #--------------- Lexing 
    #~ def errfoo(msg, a, b):
        #~ printme(msg)
        #~ sys.exit()
    #~ clex = CLexer(errfoo, lambda t: False)
    #~ clex.build()
    #~ clex.input(source_code)
    
    #~ while 1:
        #~ tok = clex.token()
        #~ if not tok: break
            
        #~ printme([tok.value, tok.type, tok.lineno, clex.filename, tok.lexpos])

    #--------------- Parsing
    source_code = r'''#line 1 "..\..\test.h"
    #line 2 "..\-\~\^\_\!\=\&test.h"
    int a;
    
    int main(int joe, float mmm) {
        k = 2;
        c = 8;
    }
    '''
    parser = CParser(lex_optimize=False, yacc_optimize=False, yacc_debug=True)
    ast = parser.parse(source_code, filename='zz')
    ast.show(showcoord=False, attrnames=True, nodenames=True)
    #~ nv=NodeVisitor()
    #~ nv.visit(ast)
    
    print (
        '' + 
        'sdf')

