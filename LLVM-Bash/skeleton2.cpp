#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;


  void select_ins(Function &F)
  {
    static LLVMContext Context;
    //LLVMContext Context=&MyGlobalContext;
    SMDiagnostic Err;
    StringRef bc_file= StringRef("test.bc");
    std::unique_ptr<llvm::Module> module = parseIRFile(bc_file, Err, Context);
    Module *Mod = module.get();
    if(!Mod)
    {
      errs()<<"fail to open the IR file\n";
      return ;
    }
    Module::iterator FunIt;
    for(FunIt=Mod->begin();FunIt!=Mod->end();++FunIt)
    {
      for(Function::iterator i=FunIt->begin(),e=FunIt->end();i!=e;i++)
      {
        for (auto j=i->begin(),f=i->end();j!=f;j++)
        {
          //BasicBlock *block = BasicBlock::Create(Context, "entry", mul_add);
          Instruction* block=&*j;
          IRBuilder<> builder(&*block);
          Value *x = ConstantInt::get(Type::getInt32Ty(Context), 3);
          Value *y = ConstantInt::get(Type::getInt32Ty(Context), 2);
          Value *z = ConstantInt::get(Type::getInt32Ty(Context), 4);
          Value *tmp = builder.CreateBinOp(Instruction::Mul, x, y, "tmp");
          Value *tmp2 = builder.CreateBinOp(Instruction::Add, tmp, z, "tmp2");
          builder.CreateRet(tmp2);
        }
      }
    }
  }
  struct SkeletonPass : public FunctionPass {
    static char ID;
    SkeletonPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
      errs() << "I saw a function called " << F.getName() << "!\n";
      return false;
    }
  };

char SkeletonPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
// static void registerSkeletonPass(const PassManagerBuilder &,
//                          legacy::PassManagerBase &PM) {
//   PM.add(new SkeletonPass());
// }
// static RegisterStandardPasses
//   RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
//                  registerSkeletonPass);


// char Hello::ID = 0;
static RegisterPass<SkeletonPass> X("hello", "Hello World Pass",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
// static llvm::RegisterStandardPasses Y(
//     llvm::PassManagerBuilder::EP_EarlyAsPossible,
//     [](const llvm::PassManagerBuilder &Builder,
//         llvm::legacy::PassManagerBase &PM) { PM.add(new Hello()); });