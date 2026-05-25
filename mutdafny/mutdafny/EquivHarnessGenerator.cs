using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Dafny;
using Microsoft.Dafny.Plugins;
using MutDafny.Mutator;
using Type = Microsoft.Dafny.Type;

namespace MutDafny;

/// <summary>
/// Generates a behavioural-equivalence verification harness for a single mutant.
///
/// Given the original program and one mutation (pos/op/arg), it isolates the
/// changed top-level member, produces both the original and mutated versions,
/// and emits a Dafny file whose only proof obligation is that the two versions
/// agree on all inputs. A single lemma named <c>EquivCheck__</c> carries the
/// claim, so verification can target it with <c>--filter-symbol EquivCheck__</c>.
///
/// Supported declarations (everything else emits a <c>.equiv.skip</c> sentinel so
/// the mutant is conservatively kept alive):
///   - function / predicate  -> clone+rename the mutated body, prove f == f__mut
///   - method with exactly one out-param, body in a small functional subset
///     (match / if-then-else / let-bindings, structural self-recursion) ->
///     transpile both versions to functions, then the same lemma form.
/// </summary>
public class EquivHarnessGenerator(string pos, string op, string? arg, ErrorReporter reporter)
    : Rewriter(reporter)
{
    private const string LemmaName = "EquivCheck__";

    private string _stem = "harness";
    private DafnyOptions _options = null!;

    public override void PreResolve(Program program) {
        _options = program.Options;
        _stem = FilenameBase(program);

        // Remove any stale outputs from a previous run so exactly one of
        // .equiv.dfy / .equiv.skip describes this mutant.
        File.Delete(_stem + ".equiv.dfy");
        File.Delete(_stem + ".equiv.skip");

        // Capture context (the original, unmutated program) before touching it.
        string contextText = Serialize(program);

        var member = FindEnclosingMember(program, pos);
        if (member == null) {
            WriteSkip("no enclosing function/method found for the mutation span");
            return;
        }

        // Pristine clone of the original member, taken before the mutation runs.
        var cloner = new Cloner();
        MemberDecl originalMember = cloner.CloneMember(member, false);

        // Apply the mutation in place (same path MutantGenerator uses). Pass the raw
        // arg (possibly null) so operators distinguish "no arg" from an empty arg.
        var mutator = new MutatorFactory(Reporter).Create(pos, op, arg);
        mutator?.Mutate(program);
        MemberDecl mutatedMember = member; // mutated in place

        try {
            string? harness = Build(contextText, originalMember, mutatedMember);
            if (harness == null) {
                WriteSkip("declaration shape is not supported by the equivalence checker");
                return;
            }
            File.WriteAllText(_stem + ".equiv.dfy", harness);
        } catch (NotSupportedException e) {
            WriteSkip(e.Message);
        } catch (Exception e) {
            WriteSkip("harness generation failed: " + e.Message);
        }
    }

    // ------------------------------------------------------------------ build

    private string? Build(string contextText, MemberDecl original, MemberDecl mutated) {
        if (original is Function fOrig && mutated is Function fMut) {
            return BuildFromFunctions(contextText, fOrig, fMut);
        }
        if (original is Method mOrig && mutated is Method mMut) {
            // Only value-returning methods with a single out-param are transpilable.
            if (mOrig.Outs.Count != 1) return null;
            var origFn = TranspileMethod(mOrig, "__orig");
            var mutFn = TranspileMethod(mMut, "__mut");
            if (origFn == null || mutFn == null) return null;
            return Assemble(contextText, new[] { origFn, mutFn },
                LemmaText(mOrig.Name, mOrig.Ins, mOrig.Req, "__orig", "__mut"));
        }
        return null;
    }

    private string BuildFromFunctions(string contextText, Function fOrig, Function fMut) {
        // The context already contains the original function under its real name,
        // so we only need to append a renamed copy of the mutated body.
        string mutText = RenameWord(PrintMember(fMut), fMut.Name, fMut.Name + "__mut");
        return Assemble(contextText, new[] { mutText },
            LemmaText(fOrig.Name, fOrig.Ins, fOrig.Req, "", "__mut"));
    }

    /// <summary>Lemma asserting the two variants agree under the original precondition.</summary>
    private string LemmaText(string name, List<Formal> ins, List<AttributedExpression> reqs,
        string origSuffix, string mutSuffix) {
        string formals = string.Join(", ", ins.Select(f => f.Name + ": " + TypeStr(f.Type)));
        string actuals = string.Join(", ", ins.Select(f => f.Name));
        var sb = new StringBuilder();
        sb.Append($"lemma {LemmaName}({formals})\n");
        foreach (var r in reqs) {
            sb.Append("  requires " + Expr(r.E) + "\n");
        }
        sb.Append($"  ensures {name}{origSuffix}({actuals}) == {name}{mutSuffix}({actuals})\n");
        sb.Append("{\n}\n");
        return sb.ToString();
    }

    private string Assemble(string contextText, IEnumerable<string> decls, string lemma) {
        var sb = new StringBuilder();
        sb.Append(contextText.TrimEnd());
        sb.Append("\n\n// ===== equivalence harness (auto-generated) =====\n\n");
        foreach (var d in decls) {
            sb.Append(d.TrimEnd());
            sb.Append("\n\n");
        }
        sb.Append(lemma);
        return sb.ToString();
    }

    // ------------------------------------------------- method -> function transpiler

    /// <summary>
    /// Rewrites a single-out-param method into a function. Returns null when the body
    /// falls outside the supported subset. Self-calls are renamed to carry the suffix.
    /// </summary>
    private string? TranspileMethod(Method m, string suffix) {
        if (m.Body == null) return null;
        string resName = m.Outs[0].Name;
        string newName = m.Name + suffix;
        string body;
        try {
            body = ToExpr(m.Body.Body, resName);
        } catch (NotSupportedException) {
            return null;
        }
        string formals = string.Join(", ", m.Ins.Select(f => f.Name + ": " + TypeStr(f.Type)));
        var sb = new StringBuilder();
        sb.Append($"function {newName}({formals}): {TypeStr(m.Outs[0].Type)}\n");
        foreach (var r in m.Req) {
            sb.Append("  requires " + Expr(r.E) + "\n");
        }
        sb.Append("{\n");
        sb.Append(Indent(body, "  "));
        sb.Append("\n}\n");
        // Rename structural self-calls (method name -> function name with suffix).
        return RenameWord(sb.ToString(), m.Name, newName);
    }

    /// <summary>
    /// Converts a statement sequence into a Dafny expression. Supports let-bindings
    /// (var decls and local reassignments), a final assignment to the out-param, an
    /// explicit return, a tail match, and a tail if-then-else. Anything else throws.
    /// </summary>
    private string ToExpr(List<Statement> stmts, string resName) {
        if (stmts.Count == 0) throw new NotSupportedException("empty branch");
        var head = stmts[0];
        var rest = stmts.GetRange(1, stmts.Count - 1);

        // var x := E;  (peephole: drop a dead store immediately overwritten by the next stmt)
        if (head is VarDeclStmt vds && vds.Locals.Count == 1 && TryRhs(vds.Assign, out var vRhs)) {
            string x = vds.Locals[0].Name;
            if (rest.Count > 0 && rest[0] is AssignStatement reassign
                && IsSingleNamedLhs(reassign, x)) {
                return ToExpr(rest, resName); // dead store removed; reassignment provides the binding
            }
            return $"var {x} := {Expr(vRhs)};\n" + ToExpr(rest, resName);
        }

        if (head is AssignStatement a && a.Lhss.Count == 1 && TryExprRhs(a, out var aRhs)) {
            string lhs = Expr(a.Lhss[0]);
            if (lhs == resName && rest.Count == 0) {
                return Expr(aRhs); // final result
            }
            if (rest.Count > 0) {
                return $"var {lhs} := {Expr(aRhs)};\n" + ToExpr(rest, resName); // local rebind (shadowing)
            }
            throw new NotSupportedException("trailing assignment to non-result variable");
        }

        if (head is ProduceStmt prod && rest.Count == 0
            && prod.Rhss is { Count: 1 } && prod.Rhss[0] is ExprRhs pr) {
            return Expr(pr.Expr); // return E;
        }

        if (head is NestedMatchStmt nms && rest.Count == 0) {
            var sb = new StringBuilder();
            sb.Append("match " + Expr(nms.Source));
            foreach (var c in nms.Cases) {
                sb.Append($"\ncase {Pat(c.Pat)} =>\n");
                sb.Append(Indent(ToExpr(c.Body, resName), "  "));
            }
            return sb.ToString();
        }

        if (head is IfStmt ifs && rest.Count == 0 && ifs.Guard != null && ifs.Els != null) {
            string thn = ToExpr(ifs.Thn.Body, resName);
            string els = ifs.Els is BlockStmt b
                ? ToExpr(b.Body, resName)
                : ToExpr(new List<Statement> { ifs.Els }, resName);
            return $"if {Expr(ifs.Guard)} then\n{Indent(thn, "  ")}\nelse\n{Indent(els, "  ")}";
        }

        throw new NotSupportedException("unsupported statement: " + head.GetType().Name);
    }

    // ---------------------------------------------------------------- patterns

    private string Pat(ExtendedPattern p) {
        switch (p) {
            case LitPattern lit:
                return Expr(lit.OrigLit);
            case IdPattern id:
                if (id.Arguments == null || id.Arguments.Count == 0) return id.Id;
                return id.Id + "(" + string.Join(", ", id.Arguments.Select(Pat)) + ")";
            case DisjunctivePattern dp:
                return string.Join(" | ", dp.Alternatives.Select(Pat));
            default:
                throw new NotSupportedException("unsupported pattern: " + p.GetType().Name);
        }
    }

    // ----------------------------------------------------------------- helpers

    private static bool TryRhs(ConcreteAssignStatement? assign, out Expression expr) {
        expr = null!;
        if (assign is AssignStatement a) return TryExprRhs(a, out expr);
        return false;
    }

    private static bool TryExprRhs(AssignStatement a, out Expression expr) {
        expr = null!;
        if (a.Rhss.Count == 1 && a.Rhss[0] is ExprRhs er) { expr = er.Expr; return true; }
        return false;
    }

    private static bool IsSingleNamedLhs(AssignStatement a, string name) {
        return a.Lhss.Count == 1 && a.Lhss[0] is NameSegment ns && ns.Name == name;
    }

    private string Expr(Expression e) => Printer.ExprToString(_options, e);

    private string TypeStr(Type t) => t.ToString();

    private string PrintMember(MemberDecl m) {
        var sw = new StringWriter();
        var pr = new Printer(sw, _options, PrintModes.Serialization);
        switch (m) {
            case Function f: pr.PrintFunction(f, 0, false); break;
            case Method me: pr.PrintMethod(me, 0, false); break;
            default: throw new NotSupportedException("cannot print member " + m.GetType().Name);
        }
        return sw.ToString();
    }

    private string Serialize(Program program) {
        var sw = new StringWriter();
        new Printer(sw, _options, PrintModes.Serialization).PrintProgram(program, false);
        return sw.ToString();
    }

    private static string RenameWord(string text, string from, string to) {
        return Regex.Replace(text, $@"\b{Regex.Escape(from)}\b", to);
    }

    private static string Indent(string text, string pad) {
        return string.Join("\n", text.Split('\n').Select(l => l.Length == 0 ? l : pad + l));
    }

    private void WriteSkip(string reason) {
        File.WriteAllText(_stem + ".equiv.skip", reason + "\n");
    }

    private string FilenameBase(Program program) {
        // Mirror MutantGenerator.StoreProgram exactly so the harness name lines up
        // with the mutant file: a null arg still takes the trailing-underscore branch.
        string stem = Path.GetFileNameWithoutExtension(program.Name);
        return stem + (arg != "" ? $"__{pos}_{op}_{arg}" : $"__{pos}_{op}");
    }

    // ------------------------------------------------- enclosing-member lookup

    private MemberDecl? FindEnclosingMember(Program program, string targetPos) {
        // Pre-resolve, Program.Modules() is empty (it reads post-resolution signatures),
        // so walk the default module's declarations directly, recursing into nested modules.
        if (!ParseSpan(targetPos, out int start, out int end)) return null;
        return SearchModule(program.DefaultModuleDef, start, end);
    }

    private MemberDecl? SearchModule(ModuleDefinition module, int start, int end) {
        foreach (var decl in module.TopLevelDecls) {
            if (decl is LiteralModuleDecl lit) {
                var nested = SearchModule(lit.ModuleDef, start, end);
                if (nested != null) return nested;
            }
            if (decl is not TopLevelDeclWithMembers withMembers) continue;
            foreach (var member in withMembers.Members) {
                if (member.StartToken == null || member.EndToken == null) continue;
                if (member.StartToken.pos <= start && end <= member.EndToken.pos) {
                    return member;
                }
            }
        }
        return null;
    }

    private static bool ParseSpan(string s, out int start, out int end) {
        start = end = 0;
        var parts = s.Split('-');
        if (parts.Length == 1) {
            if (!int.TryParse(parts[0], out start)) return false;
            end = start;
            return true;
        }
        return int.TryParse(parts[0], out start) && int.TryParse(parts[1], out end);
    }
}
