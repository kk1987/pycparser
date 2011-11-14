import c_parser
import c_ast
cudaparser = c_parser.CParser(lex_optimize=True, yacc_debug=True, yacc_optimize=False)
buf = '''
    __host__ void xx();
    __global__ void mm(int * a)
    {
        int i;
        for(i=0; i<10; i++)
        {
           a[i]+=i;
        }
    }
    void main()
    {
        int* a;
        xx();
        mm<<<1,3>>>(a);
    }
'''
t = cudaparser.parse(buf, 'test.cu', debuglevel=0)
print t.ext
print type(t.ext)
#a = c_ast.FileAST([])
"""
nodelist = []
for i in t.ext:
   if type(i) == c_ast.FuncDef:
      print i.decl, i.param_decls, i.body, i.coord
      print i.decl.quals
      if i.decl.quals==['__global__']:
#         nodelist.extend(c_ast.CU_FuncDef(i.decl, i.param_decls, i.body, i.coord))
         cuda = c_ast.CU_FuncDef(i.decl, i.param_decls, i.body, i.coord)
         nodelist.append(("cuda", cuda))

a=c_ast.FileAST(nodelist)
a.show()
"""
t.show()
for i in t.__dict__:
   print i
