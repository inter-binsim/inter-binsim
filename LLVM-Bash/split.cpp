#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;
bool num_floyd[200][200];

struct BB {
  const int INF = 1000000;

  std::string basicblockname;
  int prepathcount = 0;
  int succpathcount = 0;

  int depth = 0;
  int height = 0;

  int maxdepth = 0;
  int maxheight = 0;

  std::vector<BB*> pretable;
  std::vector<BB*> succtable;

  std::vector<BB*> backtable;
  std::vector<BB*> inbacktable;

  BB* targetBB = nullptr;

  bool is_entry = false;
  bool is_end = false;

  bool is_succ_calc = false;
  bool is_pre_calc = false;

  std::string getName(void) { return basicblockname; }

  void setName(std::string name) { basicblockname = name; }

  void addSucc(BB* b) {
    succtable.push_back(b);
         is_end = false;
    b->is_entry = false;

    is_succ_calc = false;
    is_pre_calc = false;
    b->is_pre_calc = false;
    b->is_succ_calc = false;
  }

  void countSuccPath(void) {
    if (getSuccNum() == 0)
      succpathcount = 1;
    else
      succpathcount = 0;
    for (auto& b : succtable) {
      succpathcount += b->succpathcount;
    }
    is_succ_calc = true;
  }

  void countPrePath(void) {
     if (getPreNum() == 0)
      prepathcount = 1;
    else
      prepathcount = 0;
    for (auto& b : pretable) {
      prepathcount += b->prepathcount;
       }
    is_pre_calc = true;
  }

  int getPrePath(void) { return prepathcount; }

  int getSuccPath(void) { return succpathcount; }

  int getDepth(void) { return depth; }
  int getMaxDepth(void) { return maxdepth; }
  int getHeight(void) { return height; }
  int getMaxHeight(void) { return maxheight; }

  void countDepth(void) {
    depth = INF;
    maxdepth = 0;
    for (auto& b : pretable) {
      depth = std::min(depth, b->getDepth());
             maxdepth = std::max(maxdepth, b->getMaxDepth());
    }
    if (depth == INF)
      depth = 1;
    else
      depth = depth + 1;
    if (maxdepth == 0)
      maxdepth = 1;
    else
      maxdepth = maxdepth + 1;
  }

  void countHeight(void) {
    height = INF;
    maxheight = 0;
    for (auto& b : succtable) {
      height = std::min(height, b->getHeight());
      maxheight = std::max(maxheight, b->getMaxHeight());
    }
    if (height == INF)
      height = 1;
    else
      height = height + 1;
    if (maxheight == 0)
      maxheight = 1;
    else
      maxheight = maxheight + 1;
  }

  int getMinPath(void) { return height + depth - 1; }

  int getMaxPath(void) { return maxheight + maxdepth - 1; }

  int getSuccNum(void) { return succtable.size(); }
  int getPreNum(void) { return pretable.size(); }
};

struct Func {
  std::string functionname;
  BB* entryBB;
  std::vector<BB*> bblist;
  typedef std::map<BB*, std::map<BB*, bool>> BBPairState;
  BBPairState floydBB;
  BBPairState AllEdgeBB;

  typedef std::map<BB*, int> BBCount;

  BBCount BBtoid;

  typedef std::map<BB*, bool> BBState;

  BBState visitedBB, ispreBB;
  BBCount inBB;

  bool checkConnect(BB* pre, BB* suc) {
    if (floydBB[pre][suc] == true) return true;
    return false;
  }

  struct moreImportantBB {
    bool operator()(BB* a, BB* b) {
      if( a->backtable.size() > b->backtable.size() ) return true;
      if( a->getPreNum() > b->getPreNum() ) return true;
      if( a->getSuccNum() > b->getSuccNum() ) return true;
      if( a->inbacktable.size() > b->inbacktable.size() ) return true;
      if( a->getMaxPath() > b->getMaxPath() ) return true;
      if( a->getPrePath() > b->getPrePath() ) return true;
      if( a->getSuccPath() > b->getSuccPath() ) return true;
        return false;
    }
  };
  
  void TopSortFake(void) {
    inBB.clear();
    std::vector<BB*> rem_bblist;
    for (auto& b : bblist) {
      inBB[b] = b->getPreNum();
    }
    std::priority_queue<BB*, std::vector<BB*>, moreImportantBB> q;
    if (entryBB == nullptr) {
      std::cout << "TOPSORT ERROR: entryBB is nullptr" << std::endl;
      return;
    }
    q.push(entryBB);  
    while (!q.empty()) {
      auto x = q.top();
      q.pop();
      rem_bblist.push_back(x);
      for (auto& b : x->succtable) {
        inBB[b]--;
        if (inBB[b] == 0) q.push(b);
      }
    }
    bblist = rem_bblist;
   }

  void TopSortOriginal(void) {
    std::vector<BB*> rem_bblist;
    for (auto& b : bblist) {
      inBB[b] = b->getPreNum();
    }
    std::queue<BB*> q;
    if (entryBB == nullptr) {
      std::cout << "TOPSORT ERROR: entryBB is nullptr" << std::endl;
      return;
    }
    q.push(entryBB);       while (!q.empty()) {
      auto x = q.front();
      q.pop();
      rem_bblist.push_back(x);
      for (auto& b : x->succtable) {
        inBB[b]--;
        if (inBB[b] == 0) q.push(b);
      }
    }
    bblist = rem_bblist;
  }

  void CalcAllEdge(void) {
    for (auto& b : bblist) {
      for (auto& succ : b->succtable) {
        AllEdgeBB[b][succ] = true;
      }
      for (auto& back : b->backtable) {
        AllEdgeBB[b][back] = true;
      }
    }
  }

  void CalcFloyd(void) {
    int cc = 0;
    for(auto&b: bblist) {
      BBtoid[b] = cc;
      cc ++;
    }

    for(int i = 0 ; i <bblist.size(); i++)
      for(int j = 0; j < bblist.size(); j++)
        num_floyd[i][j] = 0;

    for( int i = 0; i < bblist.size(); i++ ) {
      num_floyd[i][i] = true;
      for(auto& succ: bblist[i]->succtable) {
        num_floyd[i][BBtoid[succ]] = true;
      }
    }

    for(int i = 0 ; i < bblist.size(); i++) {
      for(int j = 0; j < bblist.size(); j ++) {
        for(int k = 0; k < bblist.size(); k++) {
          if(num_floyd[i][k] && num_floyd[k][j]) {
            num_floyd[i][j] = true;
          }
        }
      }
    }

    for(int i = 0; i < bblist.size(); i++)
      for(int j = 0; j < bblist.size(); j++)
        if(num_floyd[i][j]) {
          floydBB[bblist[i]][bblist[j]] = true;
        }
         
                            
                                                         
  }

  void setName(std::string name) { functionname = name; }

  bool checkBackConnect(BB* pre, BB* suc) {
    if (AllEdgeBB[pre][suc] == true) return true;
    return false;
  }

  void visitSucc(BB* b) {
     visitedBB[b] = true;
    ispreBB[b] = true;
    if (b->getSuccPath() != 0) return;
    for (auto& succ : b->succtable) {
      if (ispreBB[succ] == true) {
        b->backtable.push_back(succ);
        succ->inbacktable.push_back(b);
      }
      if(visitedBB[succ] == false)visitSucc(succ);
     }
           ispreBB[b] = false;
  }

  void visitSuccTwice(BB* b) {
    std::vector<BB*> rem_suc;
    visitedBB[b] = true;
    if (b->getSuccPath() != 0) return;
    for (auto& succ : b->succtable) {
      bool ok = true;
      for (auto& bac : b->backtable) {
        if(bac == succ) ok = false;
      }
      if(ok) rem_suc.push_back(succ);
      if(visitedBB[succ] == false) visitSuccTwice(succ);
    }
    b->succtable = rem_suc;
    b->countSuccPath();
    b->countHeight();
  }

  void visitPre(BB* b) {
    if (b->getPrePath() != 0) return;
    for (auto& c : b->pretable) {
      visitPre(c);
    }
    b->countPrePath();
    b->countDepth();
  }

  void CalcPath(void) {
    visitedBB.clear();
    visitSucc(entryBB);
    visitedBB.clear();
    visitSuccTwice(entryBB);
    for (auto& b : bblist) {
      for (auto& suc : b->succtable) {
        suc->pretable.push_back(b);
      }
    }
    visitedBB.clear();
    for (auto& b : bblist) {
      if (b->getSuccNum() == 0) {
        visitPre(b);
      }
    }
  }

  BB* getOrInsertBB(std::string name) {
    for (auto& b : bblist) {
      if (b->getName() == name) {
        return b;
      }
    }
    BB* b = new BB;
    b->setName(name);
    bblist.push_back(b);
    return b;
  }

  void ReadFromFile(std::string filename) {
    std::ifstream file;
    file.open(filename);
    int n = 0;
    file >> n;
         std::string bbname;
    BB *pre, *suc;
    for (int i = 0; i < n; i++) {
      file >> bbname;
      pre = getOrInsertBB(bbname);
      file >> bbname;
      suc = getOrInsertBB(bbname);

      pre->addSucc(suc);
    }
    file.close();

    this->setName(filename);
   }

  void setDefaultEntry(std::string entry_name = "1") {
    entryBB = getOrInsertBB(entry_name);
  }

  void printNode(BB* b) {
              return;
  }

  void print(void) {
    std::cout << functionname << std::endl;
    if (entryBB != nullptr) {
      printNode(entryBB);
    } else {
      std::cout << "Function have no entry BB ! " << std::endl;
    }
  }

  BB* getBB(std::string name);   };

struct BBNode {
  BB *current, *target;
  BB* mimi;
};

struct AttackBB {
  const int inf = 1000000;
  Func *current, *target;
  Func* mimi;

  AttackBB() { mimi = new Func(); }

  std::map<BB*, BBNode*> targetmatch;      std::map<BB*, BBNode*> currentmatch;     std::map<BB*, bool> targetused;

  typedef std::pair<BB*, BB*> BBEdge;
  typedef std::vector<BBEdge*> BBPath;
  typedef std::map<BB*, bool> BBState;  

  typedef std::map<BB*, BB*> BBPair;

  typedef std::vector<BB*> BBList;
  std::map<BB*, BBList*> candidateBB;

  BBState usedBB;
  BBPath cur_tar_edges;

  std::map<BB*, std::map<BB*, BBPath>> BBPathMap;

  std::map<BB*, BBState> BBMap;

  std::map<BBEdge*, BBPath*> targetPath;   
     static bool compareDepth(BB* a, BB* b) {
    return a->getDepth() < b->getDepth();
  }

         
           void CalcSingleCandidateBB(BB* currentBB, BBList* candidates) {
    for (auto& b : target->bblist) {
      if (checkBBMatch(currentBB, b)) {
        candidates->push_back(b);
      }
    }
  }

           void getCandidates(void) {
    for (auto& b : current->bblist) {
      candidateBB[b] = new BBList;
      candidateBB[b]->clear();
      CalcSingleCandidateBB(b, candidateBB[b]);
    }
  }

  bool dfsFindAllEdgePath(int x) {
    if( x == cur_tar_edges.size() ) {
      return true;
    }

    std::vector<BBPath*> *pathAll = new std::vector<BBPath*>;
    std::map<BB*, BB*> *father = new std::map<BB*, BB*>;
    BBEdge* e = cur_tar_edges[x];

    dfsAllPath(e->first, e->second, pathAll, father);
 
    bool state = false;
    for(auto &p: *pathAll) {
      if( state == true ) {
        continue;
      }
      targetPath[e] = p;
      setPath(e);
      state = dfsFindAllEdgePath(x + 1);
      if( state == true ) {
         continue; 
      }
      else {
        erasePath(e);
        continue;
      }
    }

    for(int i = 0; i < pathAll->size(); i++) {
      if ( (*pathAll)[i] != targetPath[e] ) {
        delete (*pathAll)[i];
      }
    }
    delete pathAll;
    delete father;
    return state;
  }

      bool FindEdgeMatchNoRandom() {
    usedBB.clear();
    for (auto& edge : cur_tar_edges) {
      usedBB[edge->first] = true;
      usedBB[edge->second] = true;
    }
    bool state = dfsFindAllEdgePath(0);
    return state;
  }

     bool checkBBMatch(BB* c, BB* t) {
    if (c->backtable.size() > t->backtable.size()) return false;
    if (c->inbacktable.size() > t->inbacktable.size())return false;
    if (c->getSuccPath() > t->getSuccPath()) return false;
    if (c->getPrePath() > t->getPrePath()) return false;
    if (c->getMaxPath() > t->getMaxPath()) return false;
    if (c->getSuccNum() > t->getSuccNum()) return false;
    if (c->getPreNum() > t->getPreNum())return false;
         if( (c->getSuccNum() + c->backtable.size()) == 0 && (t->getSuccNum()+t->backtable.size()) != 0 ) return false;
    return true;
  }

        void sortCandidates() {
    for (auto& b : current->bblist) {
      auto vec = candidateBB[b];
      if (vec == nullptr) {
        std::cout << "Sort wrong: empty candidates!" << std::endl;
        break;
      }
      std::sort(vec->begin(), vec->end(), compareDepth);
    }
  }

     BB* getMatchFromCurrent(BB* b) { return currentmatch[b]->target; }

     BB* getMatchFromTarget(BB* b) { return targetmatch[b]->current; }

        void setMatch(BB* currentBB, BB* targetBB) {
    if (currentmatch.count(currentBB) > 0 &&
        currentmatch[currentBB] != nullptr) {
      usedBB[getMatchFromCurrent(currentBB)] = false;
      targetmatch[getMatchFromCurrent(currentBB)] = nullptr;
      currentmatch[currentBB]->target = nullptr;
    } else {
      currentmatch[currentBB] = new BBNode;
    }
    auto node = currentmatch[currentBB];
    node->current = currentBB;
    node->target = targetBB;

    targetmatch[targetBB] = node;
    usedBB[targetBB] = true;

    for (auto& pre : currentBB->pretable) {
      BBEdge* edge = new BBEdge(getMatchFromCurrent(pre), targetBB);
      cur_tar_edges.push_back(edge);
    }
  }

  void setPath(BBEdge* edge) {
    auto path = targetPath[edge];
    if (path == nullptr) return;
    for (auto& e : *path) {
      if (e->second != edge->second) {
        usedBB[e->second] = true;
      }
    }
  }

     void erasePath(BBEdge* edge) {
    auto path = targetPath[edge];
    if (path == nullptr) return;
    for (auto& e : *path) {
      if (e->second != edge->second) {
        usedBB[e->second] = false;
      }
      delete e;
    }
    path->clear();
  }

        void eraseMatch(BB* currentBB) {
    if (currentmatch.count(currentBB) == 0 ||
        currentmatch[currentBB] == nullptr)
      return;

    auto node = currentmatch[currentBB];
    auto targetBB = getMatchFromCurrent(currentBB);

    targetmatch[targetBB] = nullptr;
    currentmatch[currentBB] = nullptr;

    delete node;

    BBPath rem_cur;
    rem_cur.clear();
    for (auto& edge : cur_tar_edges) {
      if (edge->second == targetBB) {
        erasePath(edge);
      } else {
        rem_cur.push_back(edge);
      }
    }
    cur_tar_edges = rem_cur;
  }

     bool checkConnect(BB* pre, BB* suc) { return target->checkConnect(pre, suc); }

  bool checkBackConnect(BB* pre, BB* suc) {
    return target->checkBackConnect(pre, suc);
  }

        bool checkPreConnect(BB* currentBB, BB* matchBB) {
    for (auto& b : currentBB->pretable) {
      BB* matchpre = getMatchFromCurrent(b);
      if (checkConnect(matchpre, matchBB) == false) return false;
    }
    for (auto& b : currentBB->backtable) {
      BB* matchback = getMatchFromCurrent(b);
      if (checkBackConnect(matchBB, matchback) == false) return false;
    }
    return true;
  }

     bool checkUsed(BB* matchBB) {
    if (usedBB.count(matchBB) == 0 || usedBB[matchBB] == false) {
      return false;
    }
    return true;
  }

     bool checkFinal() {       bool state;
    state = FindEdgeMatchNoRandom();
    return state;
  }

        int dfs_match_count = 0;
  bool dfsMatch(int bbid, const int list_len) {
    if (bbid == list_len) {
      return checkFinal();
    }
    int count = 0;
    for (auto& b : *(candidateBB[current->bblist[bbid]])) {
      if (checkUsed(b)) continue;
      if (!checkPreConnect(current->bblist[bbid], b)) continue;
      setMatch(current->bblist[bbid], b);
      count = count + 1;
      bool state = dfsMatch(bbid + 1, list_len);
      if (state == true) return true;
      eraseMatch(current->bblist[bbid]);
      usedBB[b] = false;
    }
    return false;
  }

     bool TryFindMatch() {
    usedBB.clear();
    return dfsMatch(0, (current->bblist).size());
  }

     void dfsAllPath(BB* x, BB* ed, std::vector<BBPath*> *pathAll, std::map<BB*, BB*>* father) {
    if (x == ed) {
      auto y = x;
      BBPath* path = new BBPath;
      while (true) {
        auto fa = (*father)[y];
        if (fa == nullptr) break;
        path->push_back(new BBEdge(fa, y));
        y = fa;
      }
      pathAll->push_back(path);
      return;
    }
    for (auto& b : x->succtable) {
      if (usedBB.count(b) != 0 && usedBB[b] == true && b != ed) continue;
      (*father)[b] = x;
      dfsAllPath(b, ed, pathAll, father);
      (*father)[b] = nullptr;
    }
  }

  std::string FindMatch(void) {       getCandidates();
    sortCandidates();
    current->TopSortFake();
    target->CalcFloyd();
    target->CalcAllEdge();
    for(auto& b: current->bblist) {
      if( candidateBB[b]->size() == 0 ) return "no-solution-candidates";
    }
    if (TryFindMatch() == false) {
      return "no-solution";
    }
    return "solution-found";
  }

  void visitNode(BB* b) {
    for (auto& succ : b->succtable) {
      BB* pre_target = currentmatch[b]->target;
      BB* suc_target = currentmatch[succ]->target;

      BBEdge* e = new BBEdge(pre_target, suc_target);
      cur_tar_edges.push_back(e);
      visitNode(succ);
    }
  }

  void CollectCurrentEdge(void) {
         if (current->entryBB == nullptr) {
      std::cout << "current entryBB is nullptr" << std::endl;
      return;
    }
    visitNode(current->entryBB);
  }

   
  void Init(void) {
    current = new Func;
    target = new Func;
         current->ReadFromFile(" ./test.txt");
    current->setDefaultEntry("entry");
         target->ReadFromFile(" ./target.txt");
    target->setDefaultEntry("entry");
    current->CalcPath();
    target->CalcPath();
/*    for (auto& b : current->bblist) {
      std::cout << b->getName() << " : " << b->getDepth() << " "
                << b->getHeight() << " " << b->getPrePath() << " "
                << b->getSuccPath() << std::endl;
    }
    for (auto& b : target->bblist) {
      std::cout << b->getName() << " : " << b->getDepth() << " "
                << b->getHeight() << " " << b->getPrePath() << " "
                << b->getSuccPath() << std::endl;
    } */
  }

  void PrintFunc(void) {
    current->print();
    target->print();
  }

  void ReadMatchFromFile(std::string filename = " ./match.txt") {
    std::ifstream file;
    file.open(filename);

    int n = 0;
    file >> n;
    std::string bbname;
    BB *fir, *sec;
    for (int i = 0; i < n; i++) {
      file >> bbname;
      fir = current->getOrInsertBB(bbname);
      file >> bbname;
      sec = target->getOrInsertBB(bbname);

      BBNode* node = new BBNode;
      node->current = fir;
      node->target = sec;

      currentmatch[fir] = node;
      targetmatch[sec] = node;
    }
    file.close();
   }

     void printBBPath(BBPath* b) {
    if (b == nullptr) {
      std::cout << "this path wrong!" << std::endl;
      return;
    }
    for (auto& edge : (*b)) {
      std::cout << edge->first->getName() << " " << edge->second->getName()
                << "\n";
    }
  }

  void printPath(void) {
    for (auto& edge : cur_tar_edges) {
      BB* current_pre = targetmatch[edge->first]->current;
      BB* current_suc = targetmatch[edge->second]->current;
      std::cout << "----------------------" << std::endl;
      std::cout << current_pre->getName() << "->" << current_suc->getName()
                << "\n";
      std::cout << edge->first->getName() << "->" << edge->second->getName()
                << "\n";
      printBBPath(targetPath[edge]);
      std::cout << "\n\n";
    }
  }

   void printMatch(void) {
    std::ofstream out_b;
    out_b.open("match.txt");
    for (auto& bb : current->bblist) {
      BB* match = currentmatch[bb]->target;
      out_b << bb->getName() << " " << match->getName() << "\n";
    }
  }
};

int main() {
  AttackBB attacker;
  attacker.Init();
  double a = clock();
  std::string res = attacker.FindMatch();
  double b = clock() - a;
  std::cout << "using" << b / CLOCKS_PER_SEC << " s" << std::endl;
  if( res == "solution-found" ) {
    attacker.printMatch();
    std::cout << "success!" << std::endl;
  }
  else {
    std::cout << "end running:" << res << std::endl;
  }
  return 0;
}