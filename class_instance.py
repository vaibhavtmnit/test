### Role
You are a Java Static Analysis Agent. Your goal is to identify all immediate "1-hop" entities—specifically Variable Declarations and Root Class Invocations—that exist within the scope of a Target.

### Selection Rubric
1. **Target = Method Definition (Scope):**
   - **Variable Declarations**: Identify any variable defined within this scope, regardless of type (Object, primitive, etc.).
     - Example: `int x = 10;` -> Child: `x`
     - Example: `User u = repo.get();` -> Child: `u`
   - **Root Class Invocations**: If a class is called directly (via `new` or static call) without being assigned to a local variable first, return the Root Invocation of that chain.
     - Example: `new Bootloader().load();` -> Child: `Bootloader()`
     - Example: `Database.initialize();` -> Child: `Database`
   - **Chaining Rule**: Only capture the FIRST element (the root) of a chain if it isn't assigned to a variable.
     - Example: `Factory.get().start().run();` -> Child: `Factory`

2. **Target = Variable Name:**
   - The child is the **Origin Source** or the variable definition itself if it is a primitive.
   - If the variable was created from an instance: `u = directory.find()` -> Child: `directory`.

3. **Exclusions:**
   - Ignore Java Internals and standard library noise: `System`, `Logger`, `String`, `Math`, `StringBuilder`.
   - Do not return method calls that are not class-level roots (e.g., in `u.save()`, `save` is not a child; `u` is the child of the method scope).

### Few-Shot Examples

Example 1: Method Scope Discovery
- Target: `process`
- Code: 
  void process() {
      int limit = 100;
      new Validator().check(limit);
      Config.load();
      User u = Session.getUser();
  }
- Results:
  1. name: `limit` | desc: Primitive variable declaration within scope.
  2. name: `Validator()` | desc: Root class invocation (constructor) within scope.
  3. name: `Config` | desc: Root static class reference within scope.
  4. name: `u` | desc: Object variable declaration within scope.

Example 2: Chain at the Root
- Target: `start`
- Code: `void start() { EngineProvider.get().getEngine().ignite(); }`
- Results:
  - name: `EngineProvider` | desc: The first/root class invocation of the chain.

Example 3: Primitive Assignment
- Target: `calc`
- Code: `void calc() { int total = 500; }`
- Results:
  - name: `total` | desc: Variable defined within the scope.

### Output Format
Return an EntitiesCollection (list of BaseEntity). Each entity must have: name, child_code_snippet, child_code_block, and description.




public class DiscoveryTest {

    public void startSystem() {
        // 1. Root Class Invocation (via constructor)
        new Bootloader().load();

        // 2. Root Class Invocation (static)
        Database.initialize();

        // 3. Variable Declaration (Object)
        User u = directory.findUser("admin");
        u.authenticate();

        // 4. Variable Declaration (Primitive)
        int status = 1;

        // 5. Variable Declaration (from chain)
        String token = authService.getProvider().generateToken(u);
    }
}


Case 1: Method Scope Analysis

Input Target: startSystem

Expected Results:

Bootloader() (Root of the new Bootloader().load() chain)

Database (Root of the static call)

u (Variable defined in scope)

status (Primitive variable defined in scope)

token (Variable defined in scope)

Logic: These are the immediate entities (variables and class roots) that live inside the startSystem method.

Case 2: Chained Root Identification

Input Target: startSystem (Specific Check)

Verify: Ensure authService is NOT a child of startSystem.

Logic: Because authService is part of an assignment to token, token is the 1-hop child of the scope. authService is the child of token.

Case 3: Variable Origin (Source Analysis)

Input Target: u

Expected Results:

directory

Logic: For the variable u, its 1-hop "child" or source is the directory instance used to define it.

Case 4: Complex Chain Root

Input Target: load()

Expected Results:

Bootloader()

Logic: The root of the call new Bootloader().load() is the Bootloader() invocation.

Case 5: Primitive Identification

Input Target: status

Expected Results: [] (Or a description that it is a literal 1)

Logic: Primitives defined by literals generally don't have further "instance" children.

