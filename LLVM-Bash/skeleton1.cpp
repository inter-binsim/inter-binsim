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
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/InlineAsm.h"
#include <vector>
#include<fstream>
using namespace std;
using namespace llvm;

//bool SkeletonPass::dfs(BasicBlock* start, BasicBlock* end);
static LLVMContext Context;
struct SkeletonPass : public FunctionPass {
    static char ID;
    SkeletonPass() : FunctionPass(ID) {}
    map<string, BasicBlock*> map_n_b;
    vector< BasicBlock * > bb_vector;
    vector< BasicBlock * > bb_fakeins;//block to insert fake instructions
    map<BasicBlock*, int> visit;
    map<string, string> map_b;
    map<string, string> map_b_r;//reverse of map_b
    map<string, int> suc_num;//map: target block, number of suc
    map<string, vector<string> > suc_str; //target func suc basic block name
    vector<pair<BasicBlock*, BasicBlock*>> cut_pair;// saving pair relation
    ReturnInst* returninst;// save return inst

    //string funcAname = "show_data";
    string funcBname = "main";

    virtual bool runOnFunction(Function &F) {
//      errs() << "I saw a function called !" << F.getName() << "!\n";
      //select_ins(F);
      //string func_name="adaline_fit_sample";//F.getName().data();
      std::ifstream in_f;
      in_f.open(" ./passsource_target.txt");
      string match_l;
      match_l.clear();
      getline(in_f, match_l);
      string func_name;
      int pos=match_l.find(" ");
      func_name=match_l.substr(0, pos);
      funcBname=match_l.substr(pos+1);
      int pos_tmp = 0;
      pos_tmp=func_name.find("@");
      if(F.getName().data()!=func_name.substr(pos_tmp+1))
      {
        errs()<<"fail to transfer!"<<"\n";
        return false;
      }
      errs() << "src:" << F.getName() << " = " << func_name << "\n";
      cfg_transfer(F);
      //insert_fakeins();
      
      errs()<<"transfer finish"<<"\n";
      return false;
    }

    void select_ins(Function &F);
    void create_line(BasicBlock* cur_bb, string target_b, int num_s, Function &F);//int num_s;//target_b successor numbers
    void cfg_transfer(Function &F);
    bool dfs(BasicBlock* start, BasicBlock* end);
    void insert_fakeins(void);
};


void SkeletonPass::select_ins(Function &F)
{
  
  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  int id=0;
  int count=0;
  int break_n=0;
  BasicBlock *cur_bb=NULL;

  vector< BasicBlock * > bb_vector;
  bb_vector.clear();
  for (bs=F.begin(), be=F.end(); bs!=be; bs++)
  {
    bb_vector.push_back(&*bs);
  }

  
  for (int i=0; i<bb_vector.size(); i++)
  {
      //BasicBlock *cur_bb=&*bs;
      //cur_bb=&*bs;
      cur_bb=bb_vector[i];
      if(!cur_bb->getTerminator())
      {
        continue;
      }
      if(isa<ReturnInst>(cur_bb->getTerminator()))
      {
        errs()<<"return ins:"<<F.getName()<<" "<<bb_vector.size()<<"\n";
        continue;
      }
      if (isa<BranchInst>(cur_bb->getTerminator()))
      {
        BranchInst*br=cast<BranchInst>(cur_bb->getTerminator());
        if (br->isUnconditional())//direct jmp
        {
          if(false)//cut line
          {
            id=0;
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(id);
            br->setSuccessor(id, new_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              phi_ins->replaceIncomingBlockWith(cur_bb, new_bb);
            }
          }
          else//add cond jmp
          {
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(0);
            cur_bb->getTerminator()->eraseFromParent();
            B.SetInsertPoint(cur_bb);
            Value* fake=ConstantInt::getFalse(Context);
            B.CreateCondBr(fake, new_bb, suc_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              //phi_ins->replaceIncomingBlockWith(cur_bb, new_bb);
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, new_bb);
            }
          }
          //new_bb->setSuccessor(0,suc_bb);
        }
        else if (br->isConditional())//conditional jmp
        {
          if(true)//cut line
          {
            id=1;
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(id);
            br->setSuccessor(id, new_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              phi_ins->replaceIncomingBlockWith(cur_bb, new_bb);
            }
          }
          //new_bb->setSuccessor(0,suc_bb);
        }




      }
      // if(isa<BranchInst>(cur_bb->getTerminator()))
      // {
      //   BranchInst*br=cast<BranchInst>(cur_bb->getTerminator());
      //   if
      // }
      }
}
void SkeletonPass::create_line(BasicBlock* cur_bb, string target_b, int num_s, Function &F)//int num_s;//target_b successor numbers
{
  pair<BasicBlock*, BasicBlock*> p1;
  
  

  //static LLVMContext Context;

  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  
  //int num_s;//to do: how to acquire num_s
  if(!cur_bb->getTerminator())//A block has no successor
  {
    if(num_s==1)//insert unconditonal branch
    {
      for(int i=0; i<num_s; i++)
      {
        //to do name[i], the i success of target_b
        //map name[i] -> suc_bb;
        string suc_name=suc_str[target_b][i];//suc name of target
        string cor_name=map_b_r[suc_name];//corrresponding source name
        BasicBlock* suc_bb=map_n_b[cor_name];
        
        B.SetInsertPoint(cur_bb);
        B.CreateBr(suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if (num_s==2)// insert conditional branch
    {
      B.SetInsertPoint(cur_bb);
      Value* fake=ConstantInt::getFalse(Context);
      B.CreateCondBr(fake, cur_bb, cur_bb);
      BranchInst*br = cast<BranchInst>(cur_bb->getTerminator());
      for(int i=0; i<num_s; i++)
      {
        //to do name[i], the i success of target_b
        //map name[i] -> suc_bb;
        string suc_name=suc_str[target_b][i];//suc name of target
        string cor_name=map_b_r[suc_name];//corrresponding source name
        BasicBlock* suc_bb=map_n_b[cor_name];
        br->setSuccessor(i, suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if(num_s==0)// match return or 0 out block
    {
      errs()<<"encounter retrun 1\n";
      B.SetInsertPoint(cur_bb);
      //Value* fake=ConstantFP::get(Type::getDoubleTy(Context), 0.0);
      //B.CreateRet(0);
      //B.CreateRet(fake);

      auto *new_inst = returninst->clone();
      cur_bb->getInstList().push_back(new_inst); 
      //new_inst->insertBefore(&*(cur_bb->end()));
      //Instruction *pi = &*cur_bb->end(); 
      //Instruction *newInst = new Instruction(...); 

      //cur_bb->getInstList().insertAfter(pi, dyn_cast<Instruction>(new_inst));
      //cur_bb->dump();
      //returninst
    }
    else// insert switch branch, to do
    {
      return;
    }
  }
  else if (isa<BranchInst>(cur_bb->getTerminator()))
  {
    BranchInst*br=cast<BranchInst>(cur_bb->getTerminator());
    if (br->isUnconditional())//direct jmp
    {
      if(num_s==1)//insert unconditonal branch
      {
        for(int i=0; i<num_s; i++)
        {
          //to do name[i], the i success of target_b
          //map name[i] -> suc_bb;
          string suc_name=suc_str[target_b][i];//suc name of target
          string cor_name=map_b_r[suc_name];//corrresponding source name
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          BasicBlock *ori_suc_bb=br->getSuccessor(i);//get original successor
          if(ori_suc_bb==suc_bb)
          {
            break;
          }
          else
          {
            br->setSuccessor(i, suc_bb);
            //to do, cur_b, ori_suc_bb save to struct, element.
            p1.first=cur_bb;
            p1.second=ori_suc_bb;
            cut_pair.push_back(p1);
            PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin());
            
            if(suc_bb->size())
            {
              PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin());
              if(phi_ins)
              {
                Value* v_tmp=phi_ins->getIncomingValue(0);
                phi_ins->addIncoming(v_tmp, cur_bb);
              }
            }
          }
          
        }
      }
      else if (num_s==2)// insert conditional branch
      {
        //BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
        BasicBlock *ori_suc_bb=br->getSuccessor(0);
        cur_bb->getTerminator()->eraseFromParent();
        B.SetInsertPoint(cur_bb);
        Value* True=ConstantInt::getTrue(Context);
        Value* fake=ConstantInt::getFalse(Context);
        //B.CreateCondBr(fake, new_bb, suc_bb);
        //B.SetInsertPoint(new_bb);
        //B.CreateBr(suc_bb);
        int flag=0;
        for(int i=0; i<num_s; i++)
        {
          //to do name[i], the i success of target_b
          //map name[i] -> suc_bb;
          string suc_name=suc_str[target_b][i];//suc name of target
          string cor_name=map_b_r[suc_name];//corrresponding source name
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          if(ori_suc_bb==suc_bb && i==0)
          {
            flag=1;
            
            // to do: suc_bb_next is the next of suc_bb, suggest to use suc_bb corpus, suc_bb should be tmp var
            string suc_name=suc_str[target_b][1];
            string cor_name=map_b_r[suc_name];
            BasicBlock* suc_bb_next=map_n_b[cor_name];

            B.CreateCondBr(True, suc_bb, suc_bb_next);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_next->begin()))//check phinode
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            break;
          }
          else if(ori_suc_bb==suc_bb && i==1)
          {
            flag=1;
            // to do: suc_bb_front is front of suc_bb
            string suc_name=suc_str[target_b][0];
            string cor_name=map_b_r[suc_name];
            BasicBlock* suc_bb_front=map_n_b[cor_name];

            B.CreateCondBr(fake, suc_bb_front, suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//check phinode
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            break;
          }
        }
        if(flag==0)
        {
          string suc_name=suc_str[target_b][0];
          string cor_name=map_b_r[suc_name];
          BasicBlock* suc_bb_front=map_n_b[cor_name];

          suc_name=suc_str[target_b][1];
          cor_name=map_b_r[suc_name];
          BasicBlock* suc_bb=map_n_b[cor_name];
          B.CreateCondBr(fake, suc_bb_front, suc_bb);
          //to do, cur_b, ori_suc_bb save to struct, element.
          p1.first=cur_bb;
          p1.second=ori_suc_bb;
          cut_pair.push_back(p1);

          if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
          {
            Value* v_tmp=phi_ins->getIncomingValue(0);
            phi_ins->addIncoming(v_tmp, cur_bb);
          }
          if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//check phinode
          {
            Value* v_tmp=phi_ins->getIncomingValue(0);
            phi_ins->addIncoming(v_tmp, cur_bb);
          }
        }
      }
      else// insert switch branch, to do
      {
        return;
      }
      //new_bb->setSuccessor(0,suc_bb);
    }
    else if (br->isConditional())//conditional jmp
    {
      if(num_s==2){
        BasicBlock *ori_suc_bb=br->getSuccessor(0);
        
        string suc_name=suc_str[target_b][1];//suc name of target
        string cor_name=map_b_r[suc_name];//corrresponding source name
        BasicBlock* suc_bb_next=map_n_b[cor_name];
        

        if(ori_suc_bb==suc_bb_next)
        {
          //to do: exchange suc_bb suc_bb_next
          suc_str[target_b][1]=suc_str[target_b][0];
          suc_str[target_b][0]=suc_name;
        }
        for(int i=0; i<num_s; i++)
        {
          //to do name[i], the i success of target_b
          //map name[i] -> suc_bb;
          string suc_name=suc_str[target_b][i];//suc name of target
          string cor_name=map_b_r[suc_name];//corrresponding source name
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          BasicBlock *ori_suc_bb=br->getSuccessor(i);//get original successor
          if(ori_suc_bb==suc_bb)//刚好匹配
          {
            continue;
          }
          else
          {

            br->setSuccessor(i, suc_bb);
            //to do, cur_b, ori_suc_bb save to struct, element.
            p1.first=cur_bb;
            p1.second=ori_suc_bb;
            cut_pair.push_back(p1);

            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
          }
        }
      }
      else if(num_s>2){// insert switch branch, to do
        return ;
      }
      //new_bb->setSuccessor(0,suc_bb);
    }
  }
  else if (isa<SwitchInst>(cur_bb->getTerminator()))
  {
    //to do
    return;
  }
  else if (isa<ReturnInst>(cur_bb->getTerminator()))
  {
    returninst=dyn_cast<ReturnInst>(cur_bb->getTerminator());
    if(num_s==1)//insert unconditonal branch
    {
      for(int i=0; i<num_s; i++)
      {
        //to do name[i], the i success of target_b
        //map name[i] -> suc_bb;
        string suc_name=suc_str[target_b][i];//suc name of target
        string cor_name=map_b_r[suc_name];//corrresponding source name
        BasicBlock* suc_bb=map_n_b[cor_name];
        
        cur_bb->getTerminator()->eraseFromParent();
        B.SetInsertPoint(cur_bb);
        B.CreateBr(suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if (num_s==2)// insert conditional branch
    {
      //BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
      //BasicBlock *ori_suc_bb=br->getSuccessor(0);
      cur_bb->getTerminator()->eraseFromParent();
      B.SetInsertPoint(cur_bb);
      Value* True=ConstantInt::getTrue(Context);
      Value* fake=ConstantInt::getFalse(Context);
      //B.CreateCondBr(fake, new_bb, suc_bb);
      //B.SetInsertPoint(new_bb);
      //B.CreateBr(suc_bb);
      
      string suc_name=suc_str[target_b][0];
      string cor_name=map_b_r[suc_name];
      BasicBlock* suc_bb_front=map_n_b[cor_name];

      suc_name=suc_str[target_b][1];
      cor_name=map_b_r[suc_name];
      BasicBlock* suc_bb=map_n_b[cor_name];
      B.CreateCondBr(fake, suc_bb_front, suc_bb);

      if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//check phinode
      {
        Value* v_tmp=phi_ins->getIncomingValue(0);
        phi_ins->addIncoming(v_tmp, cur_bb);
      }
      if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//check phinode
      {
        Value* v_tmp=phi_ins->getIncomingValue(0);
        phi_ins->addIncoming(v_tmp, cur_bb);
      }
    }
    else if(num_s==0)// match return or 0 out block
    {
      // B.SetInsertPoint(cur_bb);
      // //Value* fake=ConstantInt::getFalse(Context);
      // //B.CreateRet(0);
      // //B.CreateRetVoid();
      // Value* fake=ConstantFP::get(Type::getDoubleTy(Context), 0.0);
      // //B.CreateRet(0);
      // B.CreateRet(fake);
      
      //Value* fake=ConstantFP::get(Type::getDoubleTy(Context), 0.0);
      //B.CreateRet(0);
      //B.CreateRet(fake);
      // B.SetInsertPoint(cur_bb);
      // auto *new_inst = returninst->clone();
      // cur_bb->getInstList().push_back(new_inst); 

    }
    else// insert switch branch, to do
    {
      return;
    }

  }
  return ;
}
      
void SkeletonPass::cfg_transfer(Function &F)
{
  //static LLVMContext Context;
  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  int id=0;
  int count=0;
  int break_n=0;
  BasicBlock *cur_bb=NULL;
  map_n_b.clear();
  bb_vector.clear();
  bb_fakeins.clear();
  visit.clear();
  cut_pair.clear();
  int source_basic_num=0;
  for (bs=F.begin(), be=F.end(); bs!=be; bs++)
  {
    BasicBlock * test=&*bs;
    bb_vector.push_back(&*bs);
    map_n_b[test->getName().data()]=&*bs;
    source_basic_num++;
  }


  map_b.clear();
  map_b_r.clear();

  std::ifstream in_f;
  in_f.open(" ./passmatch.txt");//open match basic block, may need to change call
  if( ! in_f.is_open() ) {
    errs() << "[ERROR]fail to open match.txt" << "\n";
    exit(0);
  }
  
  string match_l;
  match_l.clear();
  while(getline(in_f, match_l))// record source -> target basics; and target -> source basics
  {
    string source, target;
    int pos=match_l.find(" ");
    source=match_l.substr(0,pos);
    target=match_l.substr(pos+1);
    map_b[source]=target;
    map_b_r[target]=source;
  }
  in_f.close();


  //StringRef str_ref = F.getName();
  //std::string str_suc ="printsTray"; //str_ref.data();
  std::string str_suc =funcBname;
  std::ifstream in_b_suc;
  string func_dir=" ./passfunc/";
  
  in_b_suc.open(func_dir+str_suc+ "_suc.txt");//txt: target block, number of suc
  string suc_l;
  suc_l.clear();
  suc_num.clear();
  
  int num_e=bb_vector.size();//existing block number
  getline(in_b_suc, suc_l);//read basic block number
  int pos=suc_l.find(" ");
  suc_l=suc_l.substr(0, pos);
  int num_insert=stoi(suc_l)-source_basic_num;//number of basicblocks to add
  for (int i=0; i< num_insert; i++)
  {
    BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);  
    //PHINode* phi_ins=dyn_cast <PHINode>(new_bb->begin());      
    //Instruction* block=&*j;
    IRBuilder<NoFolder> builder(new_bb);
    // builder.SetInsertPoint(new_bb);
    Value *x = ConstantInt::get(Type::getInt32Ty(Context), 3);
    Value *y = ConstantInt::get(Type::getInt32Ty(Context), 2);
    Value *z = ConstantInt::get(Type::getInt32Ty(Context), 4);
    Value *tmp2 = builder.CreateBinOp(Instruction::Add, x, y, "tmp");
    

    // 



    PHINode* phi_ins=dyn_cast <PHINode>(new_bb->begin());  
    bb_vector.push_back(new_bb);   
    bb_fakeins.push_back(new_bb);
  }

  //target_suc.clear();
  
  int count_tmp=0;//the index of created block in bb_vector
  while(getline(in_b_suc, suc_l))
  {
    string source, target;
    int pos=suc_l.find(" ");
    source=suc_l.substr(0, pos);
    string tail=suc_l.substr(pos+1);
    pos=tail.find(" ");
    target=tail.substr(0, pos);
    suc_num[source]=stoi(target);
    int i=stoi(target);
    tail=tail.substr(pos+1);
    while(i--)//insert suc name to the map
    {
      pos=tail.find(" ");
      target=tail.substr(0, pos);
      //tmp_=tail.substr(pos+1:);
      suc_str[source].push_back(target);
      tail=tail.substr(pos+1);
    }

    if (map_b_r.count(source) == 0)// establish relationship between target blocks with new created blocks
    {
      std::string str = bb_vector[num_e+count_tmp]->getName().data();
      map_b[str] = source;
      map_n_b[str] = bb_vector[num_e+count_tmp];
      count_tmp+=1;
      map_b_r[source] = str;
    }
  }
  
  // std::string entry_s = "entry";
  // errs() << "check:" << map_b[entry_s] << " " << entry_s << "\n";
  // if (map_b[entry_s] != entry_s)//somtimes entry not matches entry,  thus need to change entry
  // {
  //   BasicBlock* i_ori = map_n_b[entry_s];
  //   std::string old_entry = "old_entry";
  //   i_ori->setName(old_entry);
    
  //   BasicBlock* i_tmp;
  //   std::string target_s = map_b_r[entry_s];
  //   i_tmp = map_n_b[target_s];

  //   // for (BasicBlock::iterator DI = i_tmp->begin(); DI != i_tmp->end(); ) 
  //   // {
  //   //   Instruction *Inst = &*DI;
  //   //   Inst->eraseFromParent();
  //   //   DI++;
  //   // }


   
  //   i_tmp->removeFromParent();
  //   i_tmp->removeFromParent();
  //   for (Instruction &I : *i_tmp)
  //   {
  //     I.removeFromParent();
  //   }
  //   i_tmp->insertInto(&F, i_ori);
  //   i_tmp->setName(entry_s);
  //   IRBuilder<NoFolder> builder(i_tmp);
  //   Value *x = ConstantInt::get(Type::getInt32Ty(Context), 3);
  //   Value *y = ConstantInt::get(Type::getInt32Ty(Context), 2);
  //   Value *z = ConstantInt::get(Type::getInt32Ty(Context), 4);
  //   Value *tmp2 = builder.CreateBinOp(Instruction::Add, x, y, "tmp");


  //   errs()<<"Function first:"<<F.begin()->getName()<<"\n";
  // }



  in_b_suc.close();

  for (int i=0; i<bb_vector.size(); i++)
  {
      //BasicBlock *cur_bb=&*bs;
      //cur_bb=&*bs;
      cur_bb=bb_vector[i];
      string target_b=map_b[cur_bb->getName()];//get target B's block name
      create_line(cur_bb, target_b, suc_num[target_b], F);
  }
  
  //
  int num_recover=0;//struct node number;
  num_recover=cut_pair.size();
  for(int i=0; i<num_recover; i++)//iterate to recover lines between blocks
  {
    BasicBlock* start=cut_pair[i].first;//to do
    BasicBlock* end=cut_pair[i].second;//to do
    bool flag=false;
    flag=dfs(start, end);
  }
  
  
  

}
bool SkeletonPass::dfs(BasicBlock* start, BasicBlock* end){
  BasicBlock* tmp=start;
  int num_suc=0;//success number for current block

  //num_suc=tmp->getTerminator()->getNumSuccessors();
  if(!(tmp->getTerminator()))
  {
    return false;
  }
  else if (isa<BranchInst>(tmp->getTerminator()))
  {
    BranchInst*br = cast<BranchInst>(tmp->getTerminator());
    if (br->isUnconditional())//direct jmp
    {
      num_suc=1;
    }
    else if(br->isConditional())
    {
      num_suc=2;
    }

    Value* True=ConstantInt::getTrue(Context);
    Value* fake=ConstantInt::getFalse(Context);   
    for(int j=0; j<num_suc; j++ )
    {
      
      BasicBlock* suc = br->getSuccessor(j);
      if(suc == end)//success!
      {
        //delete blocks need to be executed
        //bb_fakeins.remove(tmp);
        auto iter = std::remove(bb_fakeins.begin(), bb_fakeins.end(), tmp);
        bb_fakeins.erase(iter, bb_fakeins.end());

        if(num_suc==2){
          if(br->getCondition()==True || br->getCondition()==fake)
          {
            if(j==0){
              br->setCondition (True);
            }
            else if(j==1){
              br->setCondition (fake);
            }
          }
        }
        return true;
      }
      else 
      {
        bool flag=false;
        if(visit[suc]==0)//to do: define visit
        {
          string b_name=suc->getName().data();
          if(b_name.find("jmp")==b_name.npos){//to do, existing block cannot be iterated
            continue;
          }
          visit[suc]=1;
          flag=dfs(suc, end);
          visit[suc]=0;
          if(flag)
          {
            //delete blocks need to be executed
            //bb_fakeins.remove(tmp);
            auto iter = std::remove(bb_fakeins.begin(), bb_fakeins.end(), tmp);
            bb_fakeins.erase(iter, bb_fakeins.end());

            if(num_suc==2){
              //errs()<<"set cond!\n";
              if(br->getCondition()==True || br->getCondition()==fake)
              {
                if(j==0){
                  br->setCondition (True);
                }
                else if(j==1){
                  br->setCondition (fake);
                }
              }
            }
            return true;
          }
          //return false;
        }

      }
    }
    return false;
    //visit[tmp]=1; 
  }
  else if (isa<SwitchInst>(tmp->getTerminator()))
  {
    //to do
    return false;
  }
    
} 

void SkeletonPass::insert_fakeins(void){
  IRBuilder<> B(Context);
  int num=0;
  BasicBlock* tmp=NULL;
  std::ifstream in_f;
  std::string filepath = " ./pass";
  in_f.open(filepath + "flag_tar.txt", ios_base::in);
  string s;
  map<string, int> len_v;//len of current block
  len_v.clear();

  map<string, int> s_pos;//start pos of current block
  s_pos.clear();
  while (getline(in_f, s))
  {
      int pos=0;
      pos=s.find(" ");
      string name= s.substr(0, pos);
      string tmp = s.substr(pos+1);
      pos=tmp.find(" ");
      string pos__;
      pos__=tmp.substr(0, pos);
      s_pos[name]=stoi(pos__);
      string length= tmp.substr(pos+1);
      int len=stoi(length);
      len_v[name]=len;
  }


  std::ofstream out_f;
  out_f.open(filepath + "flag_ori.txt", std::ios::out | std::ios::trunc);
      
  for(int i=0; i<bb_fakeins.size(); i++)
  {
    tmp=bb_fakeins[i];
    string tem_myasm = tmp->getName().data();
    string target_bname;
    target_bname.clear();
    target_bname=map_b[tem_myasm];
    int num, pos_s_n;
    num=len_v[target_bname];//the number of ins to insert;
    pos_s_n=s_pos[target_bname];
    for(int i=0; i<num; i++)//insert nop, to do, phi node?
    {
      string nop_="nop";
      StringRef myasm_start = llvm::StringRef("nop");
      string cons="~{dirflag},~{fpsr},~{flags}";
      llvm::InlineAsm *IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_start, StringRef(cons), true);
      BasicBlock* cur_bb=tmp;
      llvm::BasicBlock::iterator it=cur_bb->begin();
      //it--;

      B.SetInsertPoint(cur_bb, it);
      B.CreateCall(IA, {});
    }
    
    
    string tem_myasm_start;
    tem_myasm_start=tem_myasm+"_s:";
    
    //errs()<<"tem_myasm_start:"<<tem_myasm_start<<" "<<F.getName().data()<<"\n";
    StringRef myasm_start = llvm::StringRef(tem_myasm_start);
    string cons="~{dirflag},~{fpsr},~{flags}";
    llvm::InlineAsm *IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_start, StringRef(cons), true);
    BasicBlock* cur_bb=tmp;
    llvm::BasicBlock::iterator it=cur_bb->begin();
    //it--;

    B.SetInsertPoint(cur_bb, it);
    B.CreateCall(IA, {});


    it=cur_bb->end();
    it--;//insert before the last ins
    string tem_myasm_end;
    tem_myasm_end=tem_myasm+"_e:";
    StringRef myasm_end = llvm::StringRef(tem_myasm_end);
    IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_end, StringRef(cons), true);
    B.SetInsertPoint(cur_bb, it);
    B.CreateCall(IA, {});
    
    out_f << tem_myasm_start << " " << tem_myasm_end << " " << pos_s_n << " " << num << "\n";//print all labels name;
    

    Value* True=ConstantInt::getTrue(Context);
    if (isa<BranchInst>(tmp->getTerminator()))
    {
      BranchInst*br=cast<BranchInst>(tmp->getTerminator());
      if (br->isConditional())//conditional jmp
      {
        br->setCondition (True);
        string t_s, s_t_;
        t_s= tmp->getName().data();
        s_t_="jmp1";
        if(True)//(t_s==s_t_)
        {
          BasicBlock* i1= br->getSuccessor(0);
          BasicBlock* i2= br->getSuccessor(1);
          br->setSuccessor(0, i2);
          br->setSuccessor(1, i1);
        }
        //new_bb->setSuccessor(0,suc_bb);
      }
    }
  }
  in_f.close();
  out_f.close();
  return;
}




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