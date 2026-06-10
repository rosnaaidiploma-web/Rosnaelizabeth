const DEFAULT_FAQ_DATA = [
  {
    category: "admissions",
    questions: [
      "how do i apply",
      "application process",
      "admission requirements",
      "when does admission start",
      "what are the eligibility criteria",
      "how to submit an application"
    ],
    answer: "To apply, visit the admissions page, choose your program, and complete the online application form. You will need academic transcripts, identity proof, and any required test scores. Admission deadlines vary by program, so check the latest schedule before applying."
  },
  {
    category: "courses",
    questions: [
      "what courses are available",
      "available courses",
      "undergraduate courses",
      "postgraduate courses",
      "programs offered",
      "degree programs"
    ],
    answer: "We offer a range of programs including Computer Science, Business Administration, Engineering, Arts, and Social Sciences. Both undergraduate and postgraduate degrees are available, and each program includes practical labs, project work, and expert faculty guidance."
  },
  {
    category: "fees",
    questions: [
      "tuition fee",
      "how much is the fee",
      "fee structure",
      "scholarship",
      "financial aid",
      "cost of study"
    ],
    answer: "Tuition fees depend on the program and course level. Most undergraduate programs start from Rs.38,000 per year. Scholarships and financial aid are available for eligible students. Contact the finance office or check the fees page for exact details."
  },
  {
    category: "facilities",
    questions: [
      "campus facilities",
      "library",
      "hostel",
      "labs",
      "sports",
      "student housing"
    ],
    answer: "Our campus includes modern classrooms, a digital library, computer labs, science labs, sports fields, and student housing. We also provide medical support, counseling services, cafeterias, and strong extracurricular activities to support a well-rounded student experience."
  }
];

const synonymGroups = {
  admissions: ["admission", "apply", "application", "eligibility", "deadline", "requirements", "entry"],
  courses: ["course", "program", "undergraduate", "postgraduate", "degree", "major", "specialization"],
  fees: ["fee", "tuition", "scholarship", "financial", "aid", "cost", "payment"],
  facilities: ["campus", "library", "hostel", "dorm", "lab", "sports", "gym", "cafeteria", "housing", "health"]
};

const fallbackAnswers = [
  "Could you please share more details about your question?",
  "I can help with admissions, courses, fees, and campus facilities. What would you like to know?",
  "That is a great question. Please ask about admissions, available courses, fee structure, or facilities."
];

let faqDataset = [];
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");

function addMessage(text, sender = "bot") {
  const message = document.createElement("div");
  message.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;
  const bubble = document.createElement("div");
  bubble.className = "message-text";
  bubble.textContent = text;
  message.appendChild(bubble);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function normalizeInput(text) {
  return text.trim().toLowerCase().replace(/[?!.]/g, "");
}

function tokenize(text) {
  return normalizeInput(text).split(/\s+/).filter(Boolean);
}

function buildSearchTokens(item) {
  const questionTokens = item.questions.flatMap((question) => tokenize(question));
  const synonymTokens = synonymGroups[item.category] || [];
  return [...new Set([...questionTokens, ...synonymTokens])];
}

function scoreItem(item, terms, normalizedText) {
  const tokens = buildSearchTokens(item);
  let score = tokens.reduce((count, token) => {
    return count + (terms.includes(token) ? 1 : 0);
  }, 0);

  if (normalizedText.includes(item.category)) {
    score += 2;
  }

  item.questions.forEach((question) => {
    if (normalizedText.includes(normalizeInput(question))) {
      score += 2;
    }
  });

  return score;
}

function findBestAnswer(text) {
  const normalized = normalizeInput(text);
  const terms = tokenize(normalized);
  let bestMatch = null;
  let highestScore = 0;

  faqDataset.forEach((item) => {
    const score = scoreItem(item, terms, normalized);
    if (score > highestScore) {
      highestScore = score;
      bestMatch = item;
    }
  });

  return highestScore > 0 ? bestMatch.answer : fallbackAnswers[Math.floor(Math.random() * fallbackAnswers.length)];
}

async function loadFaqDataset() {
  try {
    const response = await fetch("faqs.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data) || data.length === 0) throw new Error("Invalid JSON dataset");
    return data;
  } catch (error) {
    console.warn("Failed to load faqs.json, using fallback dataset:", error);
    return DEFAULT_FAQ_DATA;
  }
}

async function initChatbot() {
  faqDataset = await loadFaqDataset();
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = userInput.value;
  if (!question.trim()) return;

  addMessage(question, "user");
  userInput.value = "";

  setTimeout(() => {
    const answer = findBestAnswer(question);
    addMessage(answer, "bot");
  }, 250);
});

userInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

initChatbot();
